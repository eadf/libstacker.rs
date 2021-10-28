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

/// Returns all .jpg files in a path
fn collect_files(path: &Path) -> Result<Vec<PathBuf>, StackerError> {
    Ok(std::fs::read_dir(path)?
        .into_iter()
        .filter_map(|f| f.ok())
        .filter(|f| f.path().is_file())
        .filter_map(|f| match f.path().extension() {
            Some(x) => {
                if x.to_str().is_some() && x.to_str().unwrap().to_uppercase().starts_with("JPG") {
                    Some(f.path())
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect())
}

/// Stacks, using keypoint matching, all .jpg files under `path` and returns the result as a `f32` `Mat`
pub fn keypoint_match(path: PathBuf) -> Result<Mat, StackerError> {
    let mut orb = <dyn ORB>::default()?;
    let mut first_kp: Option<Vector<KeyPoint>> = None;
    let mut first_des = Mat::default();
    let mut number_of_files: f64 = 0.0;
    let mut stacked_img = Mat::default();

    for file in collect_files(&path)? {
        number_of_files += 1.0;

        let (img, img_f) = {
            let img = imread(file.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
            let mut img_f = Mat::default();
            img.convert_to(&mut img_f, opencv::core::CV_32F, 1.0, 0.0)?;
            (img, (img_f / 255.0).into_result()?.to_mat()?)
        };

        let mut kp = Vector::<KeyPoint>::new();
        let mut des = Mat::default();
        orb.detect_and_compute(&img, &Mat::default(), &mut kp, &mut des, false)?;

        if first_kp.is_none() {
            first_kp = Some(kp);
            first_des = des;
            stacked_img = img_f;
            continue;
        }

        let matches = {
            let mut matcher = BFMatcher::new(core::NORM_HAMMING, true)?;
            matcher.add(&des)?;

            let mut matches = Vector::<core::DMatch>::new();
            matcher.match_(&first_des, /*&mut des,*/ &mut matches, &Mat::default())?;
            let mut v = matches.to_vec();
            v.sort_by(|a, b| OrderedFloat(a.distance).cmp(&OrderedFloat(b.distance)));
            Vector::<core::DMatch>::from(v)
        };

        //src_pts = np.float32([first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        let src_pts = {
            let mut pts = VectorOfPoint2f::new();
            let first_kp = first_kp.as_ref().unwrap();
            for m in matches.iter() {
                pts.push(first_kp.get(m.query_idx as usize)?.pt);
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

        stacked_img = (stacked_img + warped_image).into_result()?.to_mat()?;
    }
    stacked_img = (stacked_img / number_of_files).into_result()?.to_mat()?;
    Ok(stacked_img)
}

/// Read a image file and return COLOR_BGR2GRAY and CV_32F versions of it
fn read_grey_f32(filename: &Path) -> Result<(Mat, Mat), StackerError> {
    let img = imread(filename.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
    let mut img_f32 = Mat::default();
    img.convert_to(&mut img_f32, opencv::core::CV_32F, 1.0, 0.0)?;
    img_f32 = (img_f32 / 255.0).into_result()?.to_mat()?;

    let mut img_grey = Mat::default();
    imgproc::cvt_color(&img, &mut img_grey, imgproc::COLOR_BGR2GRAY, 0)?;
    Ok((img_grey, img_f32))
}

/// Stacks, using ECC method, all .jpg files under `path` and returns the result as a `f32` `Mat`
pub fn ecc_match(path: PathBuf) -> Result<Mat, StackerError> {
    let criteria = opencv::core::TermCriteria::new(
        TermCriteria_Type::COUNT as i32 | TermCriteria_Type::EPS as i32,
        //Specify the number of iterations. TODO: I have no idea what parameter to use
        5000,
        // Specify the threshold of the increment
        // in the correlation coefficient between two iterations
        // TODO: I have no idea what parameter to use
        1e-5,
    )?;

    let files = collect_files(&path)?;
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
