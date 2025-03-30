// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2021,2025 lacklustr@protonmail.com https://github.com/eadf

//! This library contains multi-threaded image stacking functions,
//! based on OpenCV <https://crates.io/crates/opencv> and Rayon <https://crates.io/crates/rayon>.
//!
//! Copyright (c) 2021, 2025 Eadf <lacklustr@protonmail.com>.
//! License: MIT/Apache 2.0
//!
//! The is a port of code written by Mathias Sundholm.
//! Copyright (c) 2021 <https://github.com/maitek/image_stacking>
//! License: MIT
//!
//! Read more about image alignment with OpenCV here:
//! <https://learnopencv.com/image-alignment-ecc-in-opencv-c-python>

pub use opencv;
use opencv::core::{AlgorithmHint, Point2f, Vector};
use opencv::{calib3d, core, features2d, features2d::ORB, imgcodecs, imgproc, prelude::*};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// A q&d hack allowing `opencv::Mat` objects to be `Sync`.
/// Only use this on immutable `Mat` objects.
struct UnsafeMatSyncWrapper(Mat);
unsafe impl Sync for UnsafeMatSyncWrapper {}

/// A q&d hack allowing `opencv::Vector<KeyPoints>` objects to be `Sync`.
/// Only use this on immutable `Vector<KeyPoints>` objects.
struct UnsafeVectorKeyPointSyncWrapper(core::Vector<core::KeyPoint>);
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
    PoisonError(#[from] std::sync::PoisonError<core::MatExprResult<core::MatExpr>>),
}

/// Opens an image file. Returns a (`Mat<f32>`, key-points and descriptors) tuple of the file
fn orb_detect_and_compute(
    file: &std::path::Path,
) -> Result<(Mat, core::Vector<core::KeyPoint>, Mat), StackerError> {
    let mut orb = ORB::create_def()?;
    let img = imgcodecs::imread(file.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
    let mut img_f32 = Mat::default();
    img.convert_to(&mut img_f32, opencv::core::CV_32F, 1.0, 0.0)?;
    img_f32 = (img_f32 / 255.0).into_result()?.to_mat()?;

    let mut kp = core::Vector::<core::KeyPoint>::new();
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
// no_run because the test uses files that is not present
/// ```no_run
/// # use libstacker::{prelude::*, opencv::prelude::*};
/// # fn f() -> Result<(),StackerError> {
/// let keypoint_match_img:opencv::core::Mat = keypoint_match(
///     vec!["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg"],
///     KeyPointMatchParameters {
///         method: opencv::calib3d::RANSAC,
///         ransac_reproj_threshold: 5.0,
///      })?;
/// # Ok(())}
/// ```
pub fn keypoint_match<I, P>(files: I, params: KeyPointMatchParameters) -> Result<Mat, StackerError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    let mut iter = files.into_iter();

    let first_file = iter.next().ok_or(StackerError::NotEnoughFiles)?;

    let (stacked_img, first_kp, first_des) = orb_detect_and_compute(first_file.as_ref())?;
    let first_kp = UnsafeVectorKeyPointSyncWrapper(first_kp);
    let first_des = UnsafeMatSyncWrapper(first_des);
    let stacked_img: Arc<Mutex<Mat>> = Arc::new(Mutex::new(stacked_img));

    let rest_of_the_files: Vec<PathBuf> = iter.map(|p| p.as_ref().to_path_buf()).collect();

    let number_of_files = {
        let result: Result<Vec<()>, StackerError> = rest_of_the_files
            .into_par_iter()
            .map({
                let stacked_img = stacked_img.clone();
                move |file| -> Result<(), StackerError> {
                    let first_des = &first_des;
                    let first_kp = &first_kp;
                    let (img_f, kp, des) = orb_detect_and_compute(&file)?;

                    let matches = {
                        let mut matcher = features2d::BFMatcher::create(core::NORM_HAMMING, true)?;
                        matcher.add(&des)?;

                        let mut matches = core::Vector::<core::DMatch>::new();
                        matcher.match_(&first_des.0, &mut matches, &Mat::default())?;
                        let mut v = matches.to_vec();
                        v.sort_by(|a, b| OrderedFloat(a.distance).cmp(&OrderedFloat(b.distance)));
                        core::Vector::<core::DMatch>::from(v)
                    };

                    //python: src_pts = np.float32([first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    let src_pts = {
                        let mut pts: Vector<Point2f> = Vector::with_capacity(matches.len());
                        for m in matches.iter() {
                            pts.push(first_kp.0.get(m.query_idx as usize)?.pt());
                        }
                        let mut mat = Mat::from_exact_iter(pts.into_iter())?;
                        // TODO: what to do about the reshape????? python: pts.reshape(-1, 1, 2);
                        mat.reshape_mut(2, 0)?; // 2 channels, 0 rows (auto) is this correct?
                        mat
                    };

                    //python: dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    let dst_pts = {
                        let mut pts: Vector<Point2f> = Vector::with_capacity(matches.len());
                        for m in matches.iter() {
                            pts.push(kp.get(m.train_idx as usize)?.pt());
                        }
                        let mut mat = Mat::from_exact_iter(pts.into_iter())?;
                        // TODO: what to do about the reshape????? python: pts.reshape(-1, 1, 2);
                        mat.reshape_mut(2, 0)?; // 2 channels, 0 rows (auto) is this correct?
                        mat
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
                        core::Scalar::default(),
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
                }
            })
            .collect();
        result?.len() + 1
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
fn read_grey_f32(filename: &path::Path) -> Result<(Mat, Mat), StackerError> {
    let img = imgcodecs::imread(filename.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
    let mut img_f32 = Mat::default();
    img.convert_to(&mut img_f32, opencv::core::CV_32F, 1.0, 0.0)?;
    img_f32 = (img_f32 / 255.0).into_result()?.to_mat()?;

    let mut img_grey = Mat::default();
    imgproc::cvt_color(
        &img,
        &mut img_grey,
        imgproc::COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
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

impl From<EccMatchParameters> for Result<core::TermCriteria, StackerError> {
    /// Converts from a `EccMatchParameters` to TermCriteria
    /// ```
    /// # use libstacker::{prelude::*, opencv::prelude::*, opencv::core::TermCriteria_Type};
    /// let t:Result<opencv::core::TermCriteria, StackerError> = EccMatchParameters{
    ///     motion_type: MotionType::Euclidean,
    ///     max_count: None,
    ///     epsilon: Some(0.1),
    ///     gauss_filt_size:3}.into();
    /// let t = t.unwrap();
    /// assert_eq!(t.epsilon, 0.1);
    /// assert_eq!(t.typ, TermCriteria_Type::EPS as i32);
    /// ```
    fn from(r: EccMatchParameters) -> Result<core::TermCriteria, StackerError> {
        let mut rv = core::TermCriteria::default()?;
        if let Some(max_count) = r.max_count {
            rv.typ |= core::TermCriteria_Type::COUNT as i32;
            rv.max_count = max_count;
        }
        if let Some(epsilon) = r.epsilon {
            rv.typ |= core::TermCriteria_Type::EPS as i32;
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
// no_run because the test uses files that is not present
/// ```rust,no_run
/// # use libstacker::{prelude::*, opencv::prelude::*};
/// # fn f() -> Result<(),StackerError> {
/// let ecc_match_img:opencv::core::Mat = ecc_match(
///    vec!["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg"],
///    EccMatchParameters {
///       motion_type: MotionType::Homography,
///       max_count: Some(10000),
///       epsilon: Some(1e-10),
///       gauss_filt_size: 5,
///    })?;
/// # Ok(())}
/// ```
pub fn ecc_match<I, P>(files: I, params: EccMatchParameters) -> Result<Mat, StackerError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    let mut iter = files.into_iter();
    let first_file = iter.next().ok_or(StackerError::NotEnoughFiles)?;

    let criteria = Result::<core::TermCriteria, StackerError>::from(params)?;

    let (img_grey, stacked_img) = read_grey_f32(first_file.as_ref())?;
    let first_img = UnsafeMatSyncWrapper(img_grey);
    let stacked_img: Arc<Mutex<Mat>> = Arc::new(Mutex::new(stacked_img));

    let rest_of_the_files: Vec<PathBuf> = iter.map(|p| p.as_ref().to_path_buf()).collect();

    let number_of_files = {
        let result: Result<Vec<()>, StackerError> = rest_of_the_files
            .into_par_iter()
            .map({
                let stacked_img = stacked_img.clone();
                move |file| -> Result<(), StackerError> {
                    let first_img = &first_img;
                    let (img_grey, img_f32) = read_grey_f32(&file)?;

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
                            core::Scalar::default(),
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
                            core::Scalar::default(),
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
                }
            })
            .collect();
        result?.len() + 1
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

/// Detect sharpness of an image <https://stackoverflow.com/a/7768918>
/// OpenCV port of 'LAPM' algorithm (Nayar89)
pub fn sharpness_modified_laplacian(src_mat: &Mat) -> Result<f64, StackerError> {
    let mut m = unsafe { Mat::new_rows_cols(1, 3, core::CV_64FC1)? };
    m.set_2d::<f64>(0, 0, -1.0)?;
    m.set_2d::<f64>(0, 1, 2.0)?;
    m.set_2d::<f64>(0, 2, -1.0)?;

    let g = imgproc::get_gaussian_kernel(3, -1.0, core::CV_64FC1)?;
    let mut lx = Mat::default();
    imgproc::sep_filter_2d(
        src_mat,
        &mut lx,
        core::CV_64FC1,
        &m,
        &g,
        core::Point::new(-1, -1),
        0.0,
        core::BORDER_DEFAULT,
    )?;

    let mut ly = Mat::default();
    imgproc::sep_filter_2d(
        src_mat,
        &mut ly,
        core::CV_64FC1,
        &g,
        &m,
        core::Point::new(-1, -1),
        0.0,
        core::BORDER_DEFAULT,
    )?;

    let fm = (core::abs(&lx)? + core::abs(&ly)?)
        .into_result()?
        .to_mat()?;
    Ok(*core::mean(&fm, &Mat::default())?
        .0
        .first()
        .unwrap_or(&f64::MAX))
}

/// Detect sharpness of an image <https://stackoverflow.com/a/7768918>
/// OpenCV port of 'LAPV' algorithm (Pech2000)
pub fn sharpness_variance_of_laplacian(src_mat: &Mat) -> Result<f64, StackerError> {
    let mut lap = Mat::default();
    imgproc::laplacian(
        src_mat,
        &mut lap,
        core::CV_64FC1,
        1,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;
    let mut mu = Mat::default();
    let mut sigma = Mat::default();
    opencv::core::mean_std_dev(&lap, &mut mu, &mut sigma, &Mat::default())?;
    let focus_measure = sigma.at_2d::<f64>(0, 0)?;
    Ok(focus_measure * focus_measure)
}

/// Detect sharpness of an image using Tenengrad algorithm (Krotkov86)
/// <https://stackoverflow.com/a/7768918>
///
/// # Arguments
/// * `src_grey_mat` - Input grayscale image (single channel)
/// * `k_size` - Kernel size for Sobel operator (must be 1, 3, 5, or 7)
///
/// # Returns
/// Sharpness metric (higher values indicate sharper images)
/// TODO: This function does not, yet, work as intended
pub fn sharpness_tenengrad(src_grey_mat: &Mat, k_size: i32) -> Result<f64, StackerError> {
    // Compute gradients using Sobel
    let mut gx = Mat::default();
    let mut gy = Mat::default();
    imgproc::sobel(
        src_grey_mat,
        &mut gx,
        core::CV_64FC1,
        1,  // x-order
        0,  // y-order
        k_size,
        1.0,  // scale
        0.0,  // delta
        core::BORDER_DEFAULT,
    )?;
    imgproc::sobel(
        src_grey_mat,
        &mut gy,
        core::CV_64FC1,
        0,  // x-order
        1,  // y-order
        k_size,
        1.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;

    // Compute squared gradients (gx² and gy²)
    let mut gx2 = Mat::default();
    let mut gy2 = Mat::default();
    core::multiply(&gx, &gx, &mut gx2, 1.0, -1)?;
    core::multiply(&gy, &gy, &mut gy2, 1.0, -1)?;

    // Sum the squared gradients (gx² + gy²)
    let mut sum = Mat::default();
    core::add(&gx2, &gy2, &mut sum, &core::no_array(), -1)?;

    // Compute mean of the squared gradient magnitudes
    let mean = core::mean(&sum, &core::no_array())?;
    Ok(mean[0])
}

/// Detect sharpness of an image <https://stackoverflow.com/a/7768918>
/// OpenCV port of 'GLVN' algorithm (Santos97)
pub fn sharpness_normalized_gray_level_variance(src_mat: &Mat) -> Result<f64, StackerError> {
    let mut mu = Mat::default();
    let mut sigma = Mat::default();
    opencv::core::mean_std_dev(&src_mat, &mut mu, &mut sigma, &Mat::default())?;
    let focus_measure = *sigma.at_2d::<f64>(0, 0)?;
    Ok(focus_measure * focus_measure / (*mu.at_2d::<f64>(0, 0)?))
}

/// Trait for setting value in a 2d Mat<T>
/// Todo:There must be a better way to do this
pub trait SetMValue {
    fn set_2d<T: opencv::prelude::DataType>(
        &mut self,
        row: i32,
        col: i32,
        value: T,
    ) -> Result<(), StackerError>;
}

impl SetMValue for Mat {
    #[inline]
    /// ```
    /// # use libstacker::{prelude::*, opencv::prelude::* ,opencv::prelude::MatTraitConst};
    /// let mut m = unsafe { opencv::core::Mat::new_rows_cols(1, 3, opencv::core::CV_64FC1).unwrap() };
    /// m.set_2d::<f64>(0, 0, -1.0).unwrap();
    /// m.set_2d::<f64>(0, 1, -2.0).unwrap();
    /// m.set_2d::<f64>(0, 2, -3.0).unwrap();
    /// assert_eq!(-1.0, *m.at_2d::<f64>(0,0).unwrap());
    /// assert_eq!(-2.0, *m.at_2d::<f64>(0,1).unwrap());
    /// assert_eq!(-3.0, *m.at_2d::<f64>(0,2).unwrap());
    /// ```
    fn set_2d<T: opencv::prelude::DataType>(
        &mut self,
        row: i32,
        col: i32,
        value: T,
    ) -> Result<(), StackerError> {
        let v = self.at_2d_mut::<T>(row, col)?;
        *v = value;
        Ok(())
    }
}

pub mod prelude{
    pub use super::{keypoint_match, ecc_match, EccMatchParameters, StackerError, MotionType, KeyPointMatchParameters, SetMValue};
}
