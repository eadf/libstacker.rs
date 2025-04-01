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

pub mod utils;

use backtrace::Backtrace as Backtrace2;
pub use opencv;
use opencv::core::{Point2f, Vector};
use opencv::{calib3d, core, features2d, imgcodecs, imgproc, prelude::*};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::path::PathBuf;
use thiserror::Error;
use utils::{MatExt, SetMValue};

#[derive(Error, Debug)]
pub enum StackerError {
    //#[error(transparent)]
    //CvError(#[from] opencv::Error),
    #[error("OpenCV error: {source}")]
    OpenCvError {
        source: opencv::Error,
        backtrace2: Backtrace2,
    },
    #[error("Something wrong with Arc/Mutex handling")]
    MutexError,
    #[error("Not enough files")]
    NotEnoughFiles,
    #[error("Not implemented")]
    NotImplemented,

    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error(transparent)]
    PoisonError(#[from] std::sync::PoisonError<core::MatExprResult<core::MatExpr>>),
    #[error("Invalid path encoding {0}")]
    InvalidPathEncoding(PathBuf),
    #[error("Invalid parameter(s) {0}")]
    InvalidParams(String),
    #[error("Internal error {0}")]
    ProcessingError(String),
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
/// ```no_run
/// # use libstacker::{prelude::*, opencv::prelude::*};
/// # fn f() -> Result<(),StackerError> {
/// let keypoint_match_img:opencv::core::Mat = keypoint_match(
///     vec!["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg"],
///     KeyPointMatchParameters {
///         method: opencv::calib3d::RANSAC,
///         ransac_reproj_threshold: 5.0,
///      },
///      None)?;
/// # Ok(())}
/// ```
pub fn keypoint_match<I, P>(
    files: I,
    params: KeyPointMatchParameters,
    scale_down: Option<f32>,
) -> Result<Mat, StackerError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    let files = files.into_iter().map(|p| p.as_ref().to_path_buf());
    if let Some(scale_down) = scale_down {
        keypoint_match_scale_down(files, params, scale_down)
    } else {
        keypoint_match_no_scale(files, params)
    }
    .map_err(|e| match e {
        StackerError::OpenCvError {
            source: _,
            backtrace2: ref b,
        } => {
            println!("{:?}", b.clone());
            e
        }
        _ => e,
    })
}

fn keypoint_match_no_scale<I>(
    files: I,
    params: KeyPointMatchParameters,
) -> Result<Mat, StackerError>
where
    I: IntoIterator<Item = PathBuf>,
{
    let files_vec: Vec<PathBuf> = files.into_iter().collect();

    if files_vec.is_empty() {
        return Err(StackerError::NotEnoughFiles);
    }

    // Process the first file to get reference keypoints and descriptors
    let first_img = utils::imread(&files_vec[0], imgcodecs::IMREAD_COLOR)?;
    let first_img_f32_wr =
        utils::UnsafeMatSyncWrapper(first_img.convert(opencv::core::CV_32F, 1.0 / 255.0, 0.0)?);

    // Get the image size from the first image for later use
    let img_size = first_img.size()?;

    // Get keypoints and descriptors
    let (first_kp_wr, first_des_wr) = {
        let (kp, des) = utils::orb_detect_and_compute(&first_img)?;
        (
            utils::UnsafeVectorKeyPointSyncWrapper(kp),
            utils::UnsafeMatSyncWrapper(des),
        )
    };

    // we do not need first_img anymore
    drop(first_img);

    // Combined map-reduce step: Process each image and reduce on the fly
    let result = {
        // Create a reference to our wrappers that we can move into the closure
        let first_kp_wrmv = &first_kp_wr;
        let first_des_wrmv = &first_des_wr;
        let first_img_f32_wrmv = &first_img_f32_wr;
        let files_vec_mw = &files_vec;

        (0..files_vec.len())
            .into_par_iter()
            //.into_iter()
            .try_fold(
                || None, // Initial accumulator for each thread
                move |acc: Option<utils::UnsafeMatSyncWrapper>, index| {
                    let warped_image = if index == 0 {
                        // For the first image, just use the already processed image
                        first_img_f32_wrmv.0.clone()
                    } else {
                        // Process non-first images

                        let (img_grey, img_f32) = utils::read_grey_f32(&files_vec_mw[index])?;

                        //python: src_pts = np.float32([first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

                        // Detect keypoints and compute descriptors
                        let (kp, des) = utils::orb_detect_and_compute(&img_grey)?;

                        // Match keypoints
                        let matches = {
                            let mut matcher =
                                features2d::BFMatcher::create(core::NORM_HAMMING, true)?;
                            matcher.add(&des)?;

                            let mut matches = core::Vector::<core::DMatch>::new();
                            matcher.match_(&first_des_wrmv.0, &mut matches, &Mat::default())?;
                            let mut v = matches.to_vec();
                            v.sort_by(|a, b| {
                                OrderedFloat(a.distance).cmp(&OrderedFloat(b.distance))
                            });
                            core::Vector::<core::DMatch>::from(v)
                        };

                        // Calculate source points
                        let src_pts = {
                            let mut pts: Vector<Point2f> = Vector::with_capacity(matches.len());
                            for m in matches.iter() {
                                pts.push(first_kp_wrmv.0.get(m.query_idx as usize)?.pt());
                            }
                            let mut mat = Mat::from_exact_iter(pts.into_iter())?;
                            mat.reshape_mut(2, 0)?;
                            mat
                        };

                        // Calculate destination points
                        let dst_pts = {
                            let mut pts: Vector<Point2f> = Vector::with_capacity(matches.len());
                            for m in matches.iter() {
                                pts.push(kp.get(m.train_idx as usize)?.pt());
                            }
                            let mut mat = Mat::from_exact_iter(pts.into_iter())?;
                            mat.reshape_mut(2, 0)?;
                            mat
                        };

                        // Find homography matrix
                        let h = calib3d::find_homography(
                            &dst_pts,
                            &src_pts,
                            &mut Mat::default(),
                            params.method,
                            params.ransac_reproj_threshold,
                        )?;

                        // Warp image
                        let mut warped_image = Mat::default();
                        imgproc::warp_perspective(
                            &img_f32,
                            &mut warped_image,
                            &h,
                            img_size,
                            imgproc::INTER_LINEAR,
                            core::BORDER_CONSTANT,
                            core::Scalar::default(),
                        )?;

                        warped_image
                    };

                    //python: dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    // Add to the accumulator (if accumulator is empty, just use the processed image)
                    let result = if let Some(acc) = acc {
                        let rv = utils::UnsafeMatSyncWrapper(
                            (&acc.0 + &warped_image).into_result()?.to_mat()?,
                        );
                        Some(rv)
                    } else {
                        Some(utils::UnsafeMatSyncWrapper(warped_image))
                    };

                    Ok::<Option<utils::UnsafeMatSyncWrapper>, StackerError>(result)
                },
            )
            .try_reduce(
                || None,
                |acc1, acc2| match (acc1, acc2) {
                    (Some(acc1), None) => Ok(Some(acc1)),
                    (None, Some(acc2)) => Ok(Some(acc2)),
                    (Some(acc1), Some(acc2)) => {
                        // Combine the two accumulators
                        let combined = utils::UnsafeMatSyncWrapper(
                            (&acc1.0 + &acc2.0).into_result()?.to_mat()?,
                        );
                        Ok(Some(combined))
                    }
                    _ => unreachable!(),
                },
            )
    }?;

    // Final normalization
    let final_result = if let Some(result) = result {
        (result.0 / files_vec.len() as f64)
            .into_result()?
            .to_mat()?
    } else {
        return Err(StackerError::ProcessingError(
            "Empty result after reduction".to_string(),
        ));
    };

    Ok(final_result)
}

fn keypoint_match_scale_down<I>(
    files: I,
    params: KeyPointMatchParameters,
    scale_down_to: f32,
) -> Result<Mat, StackerError>
where
    I: IntoIterator<Item = PathBuf>,
{
    let files_vec: Vec<PathBuf> = files.into_iter().collect();
    if files_vec.is_empty() {
        return Err(StackerError::NotEnoughFiles);
    }

    // Process the first file to get reference keypoints and descriptors
    let first_file = &files_vec[0];
    let first_img = utils::imread(first_file, imgcodecs::IMREAD_COLOR)?;
    let first_img_f32_wr =
        utils::UnsafeMatSyncWrapper(first_img.convert(opencv::core::CV_32F, 1.0 / 255.0, 0.0)?);

    // Get the image size from the first image for later use
    let img_size = first_img.size()?;

    // Number of files for final normalization
    let number_of_files = files_vec.len();

    // Get keypoints and descriptors
    let (first_kp_small_wr, first_des_small_wr) = {
        // Scale down the first image
        let img_small = utils::scale_image(&first_img, scale_down_to)?;

        let (kp, des) = utils::orb_detect_and_compute(&img_small)?;
        (
            utils::UnsafeVectorKeyPointSyncWrapper(kp),
            utils::UnsafeMatSyncWrapper(des),
        )
    };

    // we do not need first_img anymore
    drop(first_img);

    // Combined map-reduce step: Process each image and reduce on the fly
    let result = {
        // Create a reference to our wrappers that we can move into the closure
        let first_kp_small_wrmv = &first_kp_small_wr;
        let first_des_small_wrmv = &first_des_small_wr;
        let first_img_f32_small_wrmv = &first_img_f32_wr;

        (0..files_vec.len())
            .into_par_iter()
            //.into_iter()
            .try_fold(
                || None, // Initial accumulator for each thread
                move |acc: Option<utils::UnsafeMatSyncWrapper>, index| {
                    let warped_img = if index == 0 {
                        // For the first image, just use the already processed image
                        first_img_f32_small_wrmv.0.clone()
                    } else {
                        // Process non-first images

                        let first_des_small = first_des_small_wrmv;
                        let first_kp_small = first_kp_small_wrmv;

                        // Load and scale down the image for feature detection
                        let (img_small_grey, img_original) = {
                            let (img_grey, img_original) = utils::read_grey_f32(&files_vec[index])?;
                            let img_small_grey = utils::scale_image(&img_grey, scale_down_to)?;
                            (img_small_grey, img_original)
                        };

                        // Detect features on the scaled-down image
                        let (kp_small, des_small) = utils::orb_detect_and_compute(&img_small_grey)?;

                        // Match features
                        let matches = {
                            let mut matcher =
                                features2d::BFMatcher::create(core::NORM_HAMMING, true)?;
                            matcher.add(&des_small)?;

                            let mut matches = core::Vector::<core::DMatch>::new();

                            matcher.match_(&first_des_small.0, &mut matches, &Mat::default())?;
                            let mut v = matches.to_vec();
                            v.sort_by(|a, b| {
                                OrderedFloat(a.distance).cmp(&OrderedFloat(b.distance))
                            });
                            core::Vector::<core::DMatch>::from(v)
                        };

                        // Get corresponding points (on small images)
                        let src_pts = {
                            let mut pts: Vector<Point2f> = Vector::with_capacity(matches.len());
                            for m in matches.iter() {
                                pts.push(first_kp_small.0.get(m.query_idx as usize)?.pt());
                            }
                            let mut mat = Mat::from_exact_iter(pts.into_iter())?;
                            mat.reshape_mut(2, 0)?;
                            mat
                        };

                        let dst_pts = {
                            let mut pts: Vector<Point2f> = Vector::with_capacity(matches.len());
                            for m in matches.iter() {
                                pts.push(kp_small.get(m.train_idx as usize)?.pt());
                            }
                            let mut mat = Mat::from_exact_iter(pts.into_iter())?;
                            mat.reshape_mut(2, 0)?;
                            mat
                        };

                        // Calculate homography on small images
                        let h_small = calib3d::find_homography(
                            &dst_pts,
                            &src_pts,
                            &mut Mat::default(),
                            params.method,
                            params.ransac_reproj_threshold,
                        )?;

                        // Scale homography matrix to apply to original-sized images
                        let h = utils::adjust_homography_for_scale_f64(
                            &h_small,
                            &img_small_grey,
                            &img_original,
                        )?;

                        // Warp the original full-sized image
                        let mut warped_image = Mat::default();

                        imgproc::warp_perspective(
                            &img_original,
                            &mut warped_image,
                            &h,
                            img_size,
                            imgproc::INTER_LINEAR,
                            core::BORDER_CONSTANT,
                            core::Scalar::default(),
                        )?;

                        warped_image
                    };
                    let result = if let Some(acc) = acc {
                        let rv = utils::UnsafeMatSyncWrapper(
                            (&acc.0 + &warped_img).into_result()?.to_mat()?,
                        );
                        Some(rv)
                    } else {
                        Some(utils::UnsafeMatSyncWrapper(warped_img))
                    };

                    Ok::<Option<utils::UnsafeMatSyncWrapper>, StackerError>(result)
                },
            )
            .try_reduce(
                || None,
                |acc1, acc2| match (acc1, acc2) {
                    (Some(acc1), None) => Ok(Some(acc1)),
                    (None, Some(acc2)) => Ok(Some(acc2)),
                    (Some(acc1), Some(acc2)) => {
                        // Combine the two accumulators
                        let combined = utils::UnsafeMatSyncWrapper(
                            (&acc1.0 + &acc2.0).into_result()?.to_mat()?,
                        );
                        Ok(Some(combined))
                    }
                    _ => unreachable!(),
                },
            )
    }?;

    // Final normalization
    let final_result = if let Some(result) = result {
        (result.0 / number_of_files as f64)
            .into_result()?
            .to_mat()?
    } else {
        return Err(StackerError::ProcessingError(
            "Empty result after reduction".to_string(),
        ));
    };

    Ok(final_result)
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
///    }, None)?;
/// # Ok(())}
/// ```
pub fn ecc_match<I, P>(
    files: I,
    params: EccMatchParameters,
    scale_down: Option<f32>,
) -> Result<Mat, StackerError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    let files = files.into_iter().map(|p| p.as_ref().to_path_buf());
    if let Some(scale_down) = scale_down {
        ecc_match_scaling_down(files, params, scale_down)
    } else {
        ecc_match_no_scaling(files, params)
    }
    .map_err(|e| match e {
        StackerError::OpenCvError {
            source: _,
            backtrace2: ref b,
        } => {
            println!("{:?}", b.clone());
            e
        }
        _ => e,
    })
}

pub fn ecc_match_scaling_down<I>(
    files: I,
    params: EccMatchParameters,
    scale_down_to: f32,
) -> Result<Mat, StackerError>
where
    I: IntoIterator<Item = PathBuf>,
{
    let files_vec: Vec<PathBuf> = files.into_iter().collect();

    if files_vec.is_empty() {
        return Err(StackerError::NotEnoughFiles);
    }

    let criteria = Result::<core::TermCriteria, StackerError>::from(params)?;

    let (first_img_grey_wr, first_img_f32_wr) = {
        let (img_grey, first_img_f32) = utils::read_grey_f32(&files_vec[0])?;
        (
            utils::UnsafeMatSyncWrapper(img_grey),
            utils::UnsafeMatSyncWrapper(first_img_f32),
        )
    };

    // Load first image in color and grayscale
    let full_size = first_img_f32_wr.0.size()?;

    // Scale down the first image for ECC
    let first_img_grey_small_wr =
        utils::UnsafeMatSyncWrapper(utils::scale_image(&first_img_grey_wr.0, scale_down_to)?);
    let small_size = first_img_grey_small_wr.0.size()?;

    // Combined map-reduce step: Process each image and reduce on the fly
    let result = {
        let first_img_f32_wrmv = &first_img_f32_wr;
        let files_vec_mv = &files_vec;

        (0..files_vec.len())
            .into_par_iter()
            .try_fold(
                || None, // Initial accumulator for each thread
                move |acc: Option<utils::UnsafeMatSyncWrapper>, index| {
                    let warped_image = if index == 0 {
                        // For the first image, just use the already processed image
                        first_img_f32_wrmv.0.clone()
                    } else {
                        // Load image in both color and grayscale
                        let (img_grey_original, img_f32) =
                            utils::read_grey_f32(&files_vec_mv[index])?;
                        let first_img_grey_small = &first_img_grey_small_wr;

                        // Scale down for ECC matching
                        let img_small = utils::scale_image(&img_f32, scale_down_to)?;
                        let img_grey_small = utils::scale_image(&img_grey_original, scale_down_to)?;

                        // Initialize warp matrix based on motion type
                        let mut warp_matrix_small = if params.motion_type != MotionType::Homography
                        {
                            Mat::eye(2, 3, opencv::core::CV_32F)?.to_mat()?
                        } else {
                            Mat::eye(3, 3, opencv::core::CV_32F)?.to_mat()?
                        };

                        // Find transformation on small images
                        let _ = opencv::video::find_transform_ecc(
                            &img_grey_small,
                            &first_img_grey_small.0,
                            &mut warp_matrix_small,
                            params.motion_type as i32,
                            criteria,
                            &Mat::default(),
                            params.gauss_filt_size,
                        )?;

                        let warp_matrix = if params.motion_type != MotionType::Homography {
                            // For affine transformations, adjust the translation part of the matrix
                            let mut full_warp = warp_matrix_small.clone();

                            // Scale the translation components (third column)
                            let tx = full_warp.at_2d_mut::<f32>(0, 2)?;
                            *tx *= full_size.width as f32 / small_size.width as f32;
                            let ty = full_warp.at_2d_mut::<f32>(1, 2)?;
                            *ty *= full_size.height as f32 / small_size.height as f32;

                            full_warp
                        } else {
                            utils::adjust_homography_for_scale_f32(
                                &warp_matrix_small,
                                &img_small,
                                &img_f32,
                            )?
                        };

                        let mut warped_image = Mat::default();

                        if params.motion_type != MotionType::Homography {
                            // Use warp_affine() for Translation, Euclidean and Affine
                            imgproc::warp_affine(
                                &img_f32,
                                &mut warped_image,
                                &warp_matrix,
                                full_size,
                                imgproc::INTER_LINEAR,
                                core::BORDER_CONSTANT,
                                core::Scalar::default(),
                            )?;
                        } else {
                            // Use warp_perspective() for Homography
                            imgproc::warp_perspective(
                                &img_f32,
                                &mut warped_image,
                                &warp_matrix,
                                full_size,
                                imgproc::INTER_LINEAR,
                                core::BORDER_CONSTANT,
                                core::Scalar::default(),
                            )?;
                        }
                        warped_image
                    };

                    let result = if let Some(acc) = acc {
                        let rv = utils::UnsafeMatSyncWrapper(
                            (&acc.0 + &warped_image).into_result()?.to_mat()?,
                        );
                        Some(rv)
                    } else {
                        Some(utils::UnsafeMatSyncWrapper(warped_image))
                    };

                    Ok::<Option<utils::UnsafeMatSyncWrapper>, StackerError>(result)
                },
            )
            .try_reduce(
                || None,
                |acc1, acc2| match (acc1, acc2) {
                    (Some(acc1), None) => Ok(Some(acc1)),
                    (None, Some(acc2)) => Ok(Some(acc2)),
                    (Some(acc1), Some(acc2)) => {
                        // Combine the two accumulators
                        let combined = utils::UnsafeMatSyncWrapper(
                            (&acc1.0 + &acc2.0).into_result()?.to_mat()?,
                        );
                        Ok(Some(combined))
                    }
                    _ => unreachable!(),
                },
            )?
    };
    // Final normalization
    let final_result = if let Some(result) = result {
        (result.0 / files_vec.len() as f64)
            .into_result()?
            .to_mat()?
    } else {
        return Err(StackerError::ProcessingError(
            "Empty result after reduction".to_string(),
        ));
    };

    Ok(final_result)
}

pub fn ecc_match_no_scaling<I>(files: I, params: EccMatchParameters) -> Result<Mat, StackerError>
where
    I: IntoIterator<Item = PathBuf>,
{
    let files_vec: Vec<PathBuf> = files.into_iter().collect();

    if files_vec.is_empty() {
        return Err(StackerError::NotEnoughFiles);
    }

    let criteria = Result::<core::TermCriteria, StackerError>::from(params)?;

    let (first_img_grey_wr, first_img_f32_wr) = {
        let (img_grey, first_img_f32) = utils::read_grey_f32(&files_vec[0])?;
        (
            utils::UnsafeMatSyncWrapper(img_grey),
            utils::UnsafeMatSyncWrapper(first_img_f32),
        )
    };

    // Combined map-reduce step: Process each image and reduce on the fly
    let result = {
        let first_img_f32_wrmv = &first_img_f32_wr;
        let first_img_grey_wrmw = &first_img_grey_wr;
        let files_vec_mv = &files_vec;

        (0..files_vec.len())
            .into_par_iter()
            .try_fold(
                || None, // Initial accumulator for each thread
                move |acc: Option<utils::UnsafeMatSyncWrapper>, index| {
                    let warped_image = if index == 0 {
                        // For the first image, just use the already processed image
                        first_img_f32_wrmv.0.clone()
                    } else {
                        let (img_grey, img_f32) = utils::read_grey_f32(&files_vec_mv[index])?;

                        // s, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), first_image, M, cv2.MOTION_HOMOGRAPHY)

                        let mut warp_matrix = if params.motion_type != MotionType::Homography {
                            Mat::eye(2, 3, opencv::core::CV_32F)?.to_mat()?
                        } else {
                            Mat::eye(3, 3, opencv::core::CV_32F)?.to_mat()?
                        };

                        let _ = opencv::video::find_transform_ecc(
                            &img_grey,
                            &first_img_grey_wrmw.0,
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
                        warped_image
                    };

                    let result = if let Some(acc) = acc {
                        let rv = utils::UnsafeMatSyncWrapper(
                            (&acc.0 + &warped_image).into_result()?.to_mat()?,
                        );
                        Some(rv)
                    } else {
                        Some(utils::UnsafeMatSyncWrapper(warped_image))
                    };

                    Ok::<Option<utils::UnsafeMatSyncWrapper>, StackerError>(result)
                },
            )
            .try_reduce(
                || None,
                |acc1, acc2| match (acc1, acc2) {
                    (Some(acc1), None) => Ok(Some(acc1)),
                    (None, Some(acc2)) => Ok(Some(acc2)),
                    (Some(acc1), Some(acc2)) => {
                        // Combine the two accumulators
                        let combined = utils::UnsafeMatSyncWrapper(
                            (&acc1.0 + &acc2.0).into_result()?.to_mat()?,
                        );
                        Ok(Some(combined))
                    }
                    _ => unreachable!(),
                },
            )?
    };
    // Final normalization
    let final_result = if let Some(result) = result {
        (result.0 / files_vec.len() as f64)
            .into_result()?
            .to_mat()?
    } else {
        return Err(StackerError::ProcessingError(
            "Empty result after reduction".to_string(),
        ));
    };

    Ok(final_result)
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
        3,
        1.0,
        0.0,
        core::BORDER_REPLICATE,
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
pub fn sharpness_tenengrad(src_grey_mat: &Mat, k_size: i32) -> Result<f64, StackerError> {
    // Validate kernel size
    if ![1, 3, 5, 7].contains(&k_size) {
        return Err(StackerError::InvalidParams(
            "Kernel size must be 1, 3, 5, or 7".into(),
        ));
    }
    // Compute gradients using Sobel
    let mut gx = Mat::default();
    let mut gy = Mat::default();
    imgproc::sobel(
        src_grey_mat,
        &mut gx,
        core::CV_64FC1,
        1, // x-order
        0, // y-order
        k_size,
        1.0, // scale
        0.0, // delta
        core::BORDER_DEFAULT,
    )?;
    imgproc::sobel(
        src_grey_mat,
        &mut gy,
        core::CV_64FC1,
        0, // x-order
        1, // y-order
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
    // Convert to floating point representation
    let mut src_float = Mat::default();
    src_mat.convert_to(&mut src_float, core::CV_64FC1, 1.0, 0.0)?;

    // Compute statistics
    let mut mu = Mat::default();
    let mut sigma = Mat::default();
    core::mean_std_dev(&src_float, &mut mu, &mut sigma, &Mat::default())?;

    // Numerical safety
    let variance = sigma.at_2d::<f64>(0, 0)?.powi(2);
    let mu_value = mu.at_2d::<f64>(0, 0)?.max(f64::EPSILON);

    Ok(variance / mu_value)
}

pub mod prelude {
    pub use super::{
        EccMatchParameters, KeyPointMatchParameters, MotionType, StackerError, ecc_match,
        keypoint_match,
    };
}
