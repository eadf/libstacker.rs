//! This library contains multi-threaded image stacking functions,
//! based on OpenCV <https://crates.io/crates/opencv> and Rayon <https://crates.io/crates/rayon>.
//!
//! Copyright (c) 2021 Eadf <lacklustr@protonmail.com>.
//! License: MIT/Apache 2.0
//!
//! The is a port of code written by Mathias Sundholm.
//! Copyright (c) 2021 <https://github.com/maitek/image_stacking>
//! License: MIT
//!
//! Read more about image alignment with OpenCV here:
//! <https://learnopencv.com/image-alignment-ecc-in-opencv-c-python>

pub use opencv;
use opencv::{
    calib3d,
    core::{
        self, KeyPoint, Mat, MatExpr, MatExprResult, Scalar, TermCriteria, TermCriteria_Type,
        Vector,
    },
    features2d::{BFMatcher, ORB},
    imgcodecs::{self, imread},
    imgproc,
    prelude::*,
    types::VectorOfPoint2f,
    Result,
};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// A q&d hack allowing `opencv::Mat` objects to be `Sync`.
/// Only use this on immutable `Mat` objects.
struct UnsafeMatSyncWrapper(Mat);
unsafe impl Sync for UnsafeMatSyncWrapper {}

/// A q&d hack allowing `opencv::Vector<KeyPoints>` objects to be `Sync`.
/// Only use this on immutable `Vector<KeyPoints>` objects.
struct UnsafeVectorKeyPointSyncWrapper(Vector<KeyPoint>);
unsafe impl Sync for UnsafeVectorKeyPointSyncWrapper {}

#[derive(Error, Debug)]
pub enum StackerError {
    #[error("Something wrong with Arc/Mutex handling")]
    MutexError,
    #[error("Not enough files")]
    NotEnoughFiles,
    #[error(transparent)]
    CvError(#[from] opencv::Error),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error(transparent)]
    PoisonError(#[from] std::sync::PoisonError<MatExprResult<MatExpr>>),
}

/// Returns paths all (.jpg,jpeg,tif,png) files in a single directory (non-recursive)
pub fn collect_image_files(path: &Path) -> Result<Vec<PathBuf>, StackerError> {
    Ok(std::fs::read_dir(path)?
        .into_iter()
        .filter_map(|f| f.ok())
        .filter(|f| f.path().is_file())
        .map(|f| f.path())
        .filter(|p| p.extension().is_some() && p.extension().unwrap().to_str().is_some())
        .filter(|p| {
            let extension = p.extension().unwrap().to_str().unwrap().to_uppercase();
            extension.starts_with("JPG")
                || extension.starts_with("JPEG")
                || extension.starts_with("TIF")
                || extension.starts_with("PNG")
        })
        .collect())
}

/// Opens an image file. Returns a (`Mat<f32>`, keypoints and descriptors) tuple of the file
fn orb_detect_and_compute(file: &Path) -> Result<(Mat, Vector<KeyPoint>, Mat), StackerError> {
    let mut orb = <dyn ORB>::default()?;
    let img = imread(file.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
    let mut img_f32 = Mat::default();
    img.convert_to(&mut img_f32, opencv::core::CV_32F, 1.0, 0.0)?;
    img_f32 = (img_f32 / 255.0).into_result()?.to_mat()?;

    let mut kp = Vector::<KeyPoint>::new();
    let mut des = Mat::default();
    orb.detect_and_compute(&img, &Mat::default(), &mut kp, &mut des, false)?;
    Ok((img_f32, kp, des))
}

#[derive(Debug)]
/// Structure containing the opencv parameters needed to calculate `keypoint_match()`
pub struct KeyPointMatchParameters {
    /// parameter used in `opencv::calib3d::find_homography()`
    /// Should probably always be `opencv::calib3d::RANSAC`
    pub method: i32,
    /// parameter used in `opencv::calib3d::find_homography()`
    pub ransac_reproj_threshold: f64,
    // todo: Add border_mode: core::BORDER_CONSTANT,border_value: Scalar::default(),
    // todo: impl Default
}

/// Stacks images using the OpenCV keypoint match alignment method.
/// All `files` will be aligned and stacked together. The result is returned as a `Mat<f32>`.
/// All images will be position-matched against the first image,
/// so the first image should, preferably, be the one with best focus.
// no_run because the test uses files that might not be present
/// ```no_run
/// use libstacker::{keypoint_match, collect_image_files, KeyPointMatchParameters, StackerError};
/// use std::path::PathBuf;
/// use opencv::prelude::*;
/// # fn f() -> Result<(),StackerError> {
/// let keypoint_match_img:opencv::core::Mat = keypoint_match(
///     collect_image_files(&PathBuf::from("image_stacking_py/images"))?,
///     KeyPointMatchParameters {
///         method: opencv::calib3d::RANSAC,
///         ransac_reproj_threshold: 5.0,
///      })?;
/// # Ok(())}
/// ```
pub fn keypoint_match(
    files: Vec<PathBuf>,
    params: KeyPointMatchParameters,
) -> Result<Mat, StackerError> {
    let (first_file, rest_of_the_files) =
        files.split_first().ok_or(StackerError::NotEnoughFiles)?;

    let (stacked_img, first_kp, first_des) = orb_detect_and_compute(first_file)?;
    let first_kp = UnsafeVectorKeyPointSyncWrapper(first_kp);
    let first_des = UnsafeMatSyncWrapper(first_des);
    let stacked_img: Arc<Mutex<Mat>> = Arc::new(Mutex::new(stacked_img));

    let number_of_files = {
        let res: Result<Vec<()>, StackerError> = rest_of_the_files
            .into_par_iter()
            .map(|file| -> Result<(), StackerError> {
                let (img_f, kp, des) = orb_detect_and_compute(file)?;

                let matches = {
                    let mut matcher = BFMatcher::new(core::NORM_HAMMING, true)?;
                    matcher.add(&des)?;

                    let mut matches = Vector::<core::DMatch>::new();
                    matcher.match_(&first_des.0, &mut matches, &Mat::default())?;
                    let mut v = matches.to_vec();
                    v.sort_by(|a, b| OrderedFloat(a.distance).cmp(&OrderedFloat(b.distance)));
                    Vector::<core::DMatch>::from(v)
                };

                //src_pts = np.float32([first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                let src_pts = {
                    let mut pts = VectorOfPoint2f::new();
                    for m in matches.iter() {
                        pts.push(first_kp.0.get(m.query_idx as usize)?.pt);
                    }
                    // TODO: what to do about the reshape????? pts.reshape(-1, 1, 2);
                    Mat::from_exact_iter(pts.into_iter())?
                };

                //dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                let dst_pts = {
                    let mut pts = VectorOfPoint2f::new();
                    for m in matches.iter() {
                        pts.push(kp.get(m.train_idx as usize)?.pt);
                    }
                    // TODO: what to do about the reshape????? pts.reshape(-1, 1, 2);
                    Mat::from_exact_iter(pts.into_iter())?
                };
                let h = calib3d::find_homography(
                    &dst_pts,
                    &src_pts,
                    &mut Mat::default(),
                    params.method,
                    params.ransac_reproj_threshold,
                )?;

                let mut warped_image = Mat::default();
                imgproc::warp_perspective(
                    &img_f,
                    &mut warped_image,
                    &h,
                    img_f.size()?,
                    imgproc::INTER_LINEAR,
                    core::BORDER_CONSTANT,
                    Scalar::default(),
                )?;

                if let Ok(mut stacked_img) = stacked_img.lock() {
                    // For some reason a mutex guard won't allow (*a + &b)
                    let taken_stacked_img = std::mem::take(&mut *stacked_img);
                    *stacked_img = (taken_stacked_img + &warped_image)
                        .into_result()?
                        .to_mat()?;
                    Ok(())
                } else {
                    Err(StackerError::MutexError)
                }
            })
            .collect();
        res?.len() + 1
    };

    if let Ok(stacked_img) = Arc::try_unwrap(stacked_img) {
        if let Ok(mut stacked_img) = stacked_img.into_inner() {
            stacked_img = (stacked_img / number_of_files as f64)
                .into_result()?
                .to_mat()?;
            return Ok(stacked_img);
        }
    }
    Err(StackerError::MutexError)
}

/// Read a image file and returns a (`Mat<COLOR_BGR2GRAY>`,`Mat<CV_32F>`) tuple
fn read_grey_f32(filename: &Path) -> Result<(Mat, Mat), StackerError> {
    let img = imread(filename.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
    let mut img_f32 = Mat::default();
    img.convert_to(&mut img_f32, opencv::core::CV_32F, 1.0, 0.0)?;
    img_f32 = (img_f32 / 255.0).into_result()?.to_mat()?;

    let mut img_grey = Mat::default();
    imgproc::cvt_color(&img, &mut img_grey, imgproc::COLOR_BGR2GRAY, 0)?;
    Ok((img_grey, img_f32))
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MotionType {
    Homography = opencv::video::MOTION_HOMOGRAPHY as isize,
    Affine = opencv::video::MOTION_AFFINE as isize,
    Euclidean = opencv::video::MOTION_EUCLIDEAN as isize,
    Translation = opencv::video::MOTION_TRANSLATION as isize,
}

#[derive(Debug, Copy, Clone)]
/// Structure containing the opencv parameters needed to calculate `ecc_match()`
pub struct EccMatchParameters {
    pub motion_type: MotionType,
    /// parameter used as `opencv::core::TermCriteria::max_count`
    pub max_count: Option<i32>,
    /// parameter used as `opencv::core::TermCriteria::epsilon`
    pub epsilon: Option<f64>,
    /// parameter used in `opencv::video::find_transform_ecc()`
    pub gauss_filt_size: i32,
    // todo: Add border_mode: core::BORDER_CONSTANT & border_value: Scalar::default(),
    // todo: impl Default
}

impl From<EccMatchParameters> for Result<TermCriteria, StackerError> {
    /// Converts from a `EccMatchParameters` to TermCriteria
    /// ```
    /// # use libstacker::{EccMatchParameters, MotionType, StackerError};
    /// # use opencv::core::{TermCriteria, TermCriteria_Type};
    /// let t:Result<TermCriteria, StackerError> = EccMatchParameters{
    ///     motion_type: MotionType::Euclidean,
    ///     max_count: None,
    ///     epsilon: Some(0.1),
    ///     gauss_filt_size:3}.into();
    /// let t = t.unwrap();
    /// assert_eq!(t.epsilon, 0.1);
    /// assert_eq!(t.typ, TermCriteria_Type::EPS as i32);
    /// ```
    fn from(r: EccMatchParameters) -> Result<TermCriteria, StackerError> {
        let mut rv = TermCriteria::default()?;
        if let Some(max_count) = r.max_count {
            rv.typ |= TermCriteria_Type::COUNT as i32;
            rv.max_count = max_count;
        }
        if let Some(epsilon) = r.epsilon {
            rv.typ |= TermCriteria_Type::EPS as i32;
            rv.epsilon = epsilon;
        }
        Ok(rv)
    }
}

/// Stacks images using the OpenCV ECC alignment method.
/// <https://learnopencv.com/image-alignment-ecc-in-opencv-c-python>
/// All `files` will be aligned and stacked together. The result is returned as a `Mat<f32>`.
/// All images will be position-matched against the first image,
/// so the first image should, preferably, be the one with best focus.
// no_run because the test uses files that might not be present
/// ```no_run
/// use libstacker::{ecc_match, collect_image_files, EccMatchParameters, StackerError, MotionType};
/// use std::path::PathBuf;
/// use opencv::prelude::*;
/// # fn f() -> Result<(),StackerError> {
/// let ecc_match_img:opencv::core::Mat = ecc_match(
///    collect_image_files(&PathBuf::from("image_stacking_py/images"))?,
///    EccMatchParameters {
///       motion_type: MotionType::Homography,
///       max_count: Some(10000),
///       epsilon: Some(1e-10),
///       gauss_filt_size: 5,
///    })?;
/// # Ok(())}
/// ```
pub fn ecc_match(files: Vec<PathBuf>, params: EccMatchParameters) -> Result<Mat, StackerError> {
    let criteria = Result::<TermCriteria, StackerError>::from(params)?;

    let (first_file, rest_of_the_files) =
        files.split_first().ok_or(StackerError::NotEnoughFiles)?;
    let (first_img, stacked_img) = read_grey_f32(first_file)?;
    let first_img = UnsafeMatSyncWrapper(first_img);
    let stacked_img: Arc<Mutex<Mat>> = Arc::new(Mutex::new(stacked_img));

    let number_of_files = {
        let res: Result<Vec<()>, StackerError> = rest_of_the_files
            .into_par_iter()
            .map(|file| -> Result<(), StackerError> {
                let (img_grey, img_f32) = read_grey_f32(file)?;

                // s, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), first_image, M, cv2.MOTION_HOMOGRAPHY)

                let mut warp_matrix = if params.motion_type != MotionType::Homography {
                    Mat::eye(2, 3, opencv::core::CV_32F)?.to_mat()?
                } else {
                    Mat::eye(3, 3, opencv::core::CV_32F)?.to_mat()?
                };

                let _ = opencv::video::find_transform_ecc(
                    &img_grey,
                    &first_img.0,
                    &mut warp_matrix,
                    params.motion_type as i32,
                    criteria,
                    &Mat::default(),
                    params.gauss_filt_size,
                )?;
                let mut warped_image = Mat::default();

                if params.motion_type != MotionType::Homography {
                    // Use warp_affine() for Translation, Euclidean and Affine
                    imgproc::warp_affine(
                        &img_f32,
                        &mut warped_image,
                        &warp_matrix,
                        img_f32.size()?,
                        imgproc::INTER_LINEAR,
                        core::BORDER_CONSTANT,
                        Scalar::default(),
                    )?;
                } else {
                    // Use warp_perspective() for Homography
                    // image = cv2.warpPerspective(image, M, (h, w))
                    imgproc::warp_perspective(
                        &img_f32,
                        &mut warped_image,
                        &warp_matrix,
                        img_f32.size()?,
                        imgproc::INTER_LINEAR,
                        core::BORDER_CONSTANT,
                        Scalar::default(),
                    )?;
                }
                if let Ok(mut stacked_img) = stacked_img.lock() {
                    // For some reason a mutex guard won't allow (*a + &b)
                    let taken_stacked_img = std::mem::take(&mut *stacked_img);
                    *stacked_img = (taken_stacked_img + &warped_image)
                        .into_result()?
                        .to_mat()?;
                    Ok(())
                } else {
                    Err(StackerError::MutexError)
                }
            })
            .collect();
        res?.len() + 1
    };
    if let Ok(stacked_img) = Arc::try_unwrap(stacked_img) {
        if let Ok(mut stacked_img) = stacked_img.into_inner() {
            stacked_img = (stacked_img / number_of_files as f64)
                .into_result()?
                .to_mat()?;
            return Ok(stacked_img);
        }
    }
    Err(StackerError::MutexError)
}
