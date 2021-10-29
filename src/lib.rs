use opencv::core::{MatExpr, MatExprResult};
use opencv::hub_prelude::*;
use opencv::prelude::*;
use opencv::{
    calib3d,
    core::{self, KeyPoint, Mat, Scalar, TermCriteria_Type, Vector},
    features2d::{BFMatcher, ORB},
    imgcodecs,
    imgcodecs::imread,
    imgproc,
    types::VectorOfPoint2f,
    Result,
};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// A total hack allowing opencv::Mat objects to be Sync.
/// Only use this on immutable Mat objects.
struct UnsafeMatSyncWrapper(Mat);
unsafe impl Sync for UnsafeMatSyncWrapper {}

/// A total hack allowing opencv::Vector<KeyPoints> objects to be Sync.
/// Only use this on immutable Vector objects.
struct UnsafeVecSyncWrapper(Vector<KeyPoint>);
unsafe impl Sync for UnsafeVecSyncWrapper {}

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

/// Returns all (.jpg,jpeg,tif,png) files in a path
pub fn collect_image_files(path: &Path) -> Result<Vec<PathBuf>, StackerError> {
    Ok(std::fs::read_dir(path)?
        .into_iter()
        .filter_map(|f| f.ok())
        .filter(|f| f.path().is_file())
        .filter_map(|f| match f.path().extension() {
            Some(extension) => {
                if let Some(extension) = extension.to_str() {
                    let extension = extension.to_uppercase();
                    if extension.starts_with("JPG")
                        || extension.starts_with("JPEG")
                        || extension.starts_with("TIF")
                        || extension.starts_with("PNG")
                    {
                        Some(f.path())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect())
}

/// opens an image file, returns a Mat<f32>, keypoints and descriptors of the file
fn orb_image(file: &Path) -> Result<(Mat, Vector<KeyPoint>, Mat), StackerError> {
    let mut orb = <dyn ORB>::default()?;
    let img = imread(file.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
    let mut img_f = Mat::default();
    img.convert_to(&mut img_f, opencv::core::CV_32F, 1.0, 0.0)?;
    img_f = (img_f / 255.0).into_result()?.to_mat()?;

    let mut kp = Vector::<KeyPoint>::new();
    let mut des = Mat::default();
    orb.detect_and_compute(&img, &Mat::default(), &mut kp, &mut des, false)?;
    Ok((img_f, kp, des))
}

/// Stacks, using keypoint matching, all the `files` and returns the result as a `Mat<f32>`
/// The first file will be used as the template that the other files are matched against.
/// This should be the image with best focus.
pub fn keypoint_match(files: Vec<PathBuf>) -> Result<Mat, StackerError> {
    #[allow(clippy::unnecessary_lazy_evaluations)]
    let (first_file, rest_of_the_files) = files
        .split_first()
        .ok_or_else(|| StackerError::NotEnoughFiles)?;

    let (stacked_img, first_kp, first_des) = orb_image(first_file)?;
    let first_kp = UnsafeVecSyncWrapper(first_kp);
    let first_des = UnsafeMatSyncWrapper(first_des);
    let stacked_img: Arc<Mutex<Mat>> = Arc::new(Mutex::new(stacked_img));

    let number_of_files = rest_of_the_files
        .into_par_iter()
        .map(|file| -> Result<(), StackerError> {
            let (img_f, kp, des) = orb_image(file)?;

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
                calib3d::RANSAC,
                5.0,
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
                // For some reason a mutex guard that won't allow (*a + &b)
                let taken_stacked_img = std::mem::take(&mut *stacked_img);
                *stacked_img = (taken_stacked_img + &warped_image)
                    .into_result()?
                    .to_mat()?;
                Ok(())
            } else {
                Err(StackerError::MutexError)
            }
        })
        .count()
        + 1;

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

/// Read a image file and return Mat<COLOR_BGR2GRAY> and Mat<CV_32F> versions of it
fn read_grey_f32(filename: &Path) -> Result<(Mat, Mat), StackerError> {
    let img = imread(filename.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
    let mut img_f32 = Mat::default();
    img.convert_to(&mut img_f32, opencv::core::CV_32F, 1.0, 0.0)?;
    img_f32 = (img_f32 / 255.0).into_result()?.to_mat()?;

    let mut img_grey = Mat::default();
    imgproc::cvt_color(&img, &mut img_grey, imgproc::COLOR_BGR2GRAY, 0)?;
    Ok((img_grey, img_f32))
}

/// Stacks, using ECC method, all `files` and returns the result as a Mat<f32>`
/// The first file will be used as the template that the other files are matched against.
/// This should be the image with best focus.
pub fn ecc_match(files: Vec<PathBuf>) -> Result<Mat, StackerError> {
    let criteria = opencv::core::TermCriteria::new(
        TermCriteria_Type::COUNT as i32 | TermCriteria_Type::EPS as i32,
        //Specify the number of iterations. TODO: I have no idea what parameter to use
        5000,
        // Specify the threshold of the increment
        // in the correlation coefficient between two iterations
        // TODO: I have no idea what parameter to use
        1e-5,
    )?;

    #[allow(clippy::unnecessary_lazy_evaluations)]
    let (first_file, rest_of_the_files) = files
        .split_first()
        .ok_or_else(|| StackerError::NotEnoughFiles)?;
    let (first_img, stacked_img) = read_grey_f32(first_file)?;
    let first_img = UnsafeMatSyncWrapper(first_img);
    let stacked_img: Arc<Mutex<Mat>> = Arc::new(Mutex::new(stacked_img));

    let number_of_files = rest_of_the_files
        .into_par_iter()
        .map(|file| -> Result<(), StackerError> {
            let (img_grey, img_f32) = read_grey_f32(file)?;

            // s, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), first_image, M, cv2.MOTION_HOMOGRAPHY)
            let mut warp_matrix = Mat::default();

            let _ = opencv::video::find_transform_ecc(
                &img_grey,
                &first_img.0,
                &mut warp_matrix,
                opencv::video::MOTION_HOMOGRAPHY,
                criteria,
                &Mat::default(),
                // TODO: I have no idea what parameter to use
                1,
            )?;

            // image = cv2.warpPerspective(image, M, (h, w))
            let mut warped_image = Mat::default();
            imgproc::warp_perspective(
                &img_f32,
                &mut warped_image,
                &warp_matrix,
                img_f32.size()?,
                opencv::video::MOTION_HOMOGRAPHY,
                core::BORDER_CONSTANT,
                Scalar::default(),
            )?;
            if let Ok(mut stacked_img) = stacked_img.lock() {
                // For some reason a mutex guard that won't allow (*a + &b)
                let taken_stacked_img = std::mem::take(&mut *stacked_img);
                *stacked_img = (taken_stacked_img + &warped_image)
                    .into_result()?
                    .to_mat()?;
                Ok(())
            } else {
                Err(StackerError::MutexError)
            }
        })
        .count()
        + 1;
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
