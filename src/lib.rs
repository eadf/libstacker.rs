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

pub use opencv;
use opencv::core::{Point2f, Vector};
use opencv::{calib3d, core, features2d, imgcodecs, imgproc, prelude::*};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::path::PathBuf;
use thiserror::Error;
use utils::SetMValue;

#[derive(Error, Debug)]
pub enum StackerError {
    #[error(transparent)]
    OpenCvError(#[from] opencv::Error),
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

/// Parameters for keypoint matching and homography estimation.
#[derive(Debug, Clone, Copy)]
pub struct KeyPointMatchParameters {
    /// Method used in `opencv::calib3d::find_homography()`, typically `opencv::calib3d::RANSAC`.
    pub method: i32,

    /// Reprojection threshold for RANSAC in `find_homography()`.
    /// A lower value makes RANSAC stricter (fewer matches kept), while a higher value is more lenient.
    pub ransac_reproj_threshold: f64,

    /// Ratio of best matches to keep after sorting by distance.
    /// Common values range from 0.5 to 0.8.
    pub match_keep_ratio: f32,

    /// Lowe’s ratio test threshold: how similar the best and second-best matches must be.
    /// Used in `knn_match()` to filter ambiguous matches.
    /// Common values range from 0.7 to 0.9.
    pub match_ratio: f32,

    /// Border mode used when warping images.
    /// Default: `opencv::core::BORDER_CONSTANT`.
    pub border_mode: i32,

    /// Border value used in warping.
    /// Default: `opencv::core::Scalar::default()`.
    pub border_value: core::Scalar,
}

/// Aligns and combines multiple images by matching keypoints, with optional scaling for performance.
///
/// This function processes a sequence of images by:
/// 1. Detecting ORB features (either at full resolution or scaled-down for faster processing)
/// 2. Matching features between the first (reference) image and subsequent images
/// 3. Computing homographies to align images to the reference
/// 4. Warping images to the reference frame and averaging them
///
/// When scaling is enabled:
/// - Feature detection/matching occurs on smaller images for performance
/// - Homographies are scaled appropriately for full-size warping
///
/// # Parameters
/// - `files`: An iterator of paths to image files (any type implementing `AsRef<Path>`)
/// - `params`: Configuration parameters for keypoint matching:
///   - `method`: Homography computation method (e.g., RANSAC, LMEDS)
///   - `ransac_reproj_threshold`: Maximum reprojection error for inlier classification
/// - `scale_down_width`: Controls performance/accuracy trade-off:
///   - `Some(width)`: Process features at specified width (faster, recommended 400-800px)
///   - `None`: Process at full resolution (more accurate but slower)
///   - Must be smaller than original image width when specified
///
/// # Returns
/// - `Ok((i32, Mat)).`: number of dropped frames + Combined/averaged image in CV_32F format (normalized 0-1 range)
/// - `Err(StackerError)` on failure cases:
///   - No input files provided
///   - Invalid scale width (when specified)
///   - Image loading/processing failures
///
/// # Performance Characteristics
/// - Parallel processing (Rayon) for multi-image alignment
/// - Memory efficient streaming processing
/// - Output matches size/coordinates of first (reference) image
/// - Scaling typically provides 2-4x speedup with minimal accuracy loss
///
/// ```rust,no_run
/// # use libstacker::{prelude::*, opencv::prelude::*};
/// # fn f() -> Result<(),StackerError> {
/// let keypoint_match_img:opencv::core::Mat = keypoint_match(
///     vec!["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg"],
///     KeyPointMatchParameters {
///         method: opencv::calib3d::RANSAC,
///         ransac_reproj_threshold: 5.0,
///         match_ratio: 0.9,
///          match_keep_ratio: 0.80,
///          border_mode: opencv::core::BORDER_CONSTANT,
///          border_value: opencv::core::Scalar::default(),
///      },
///      None)?.1;
/// # Ok(())}
/// ```
/// # References
/// - [OpenCV Keypoint Match](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
/// - [ORB Feature Matching](https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html)
pub fn keypoint_match<I, P>(
    files: I,
    params: KeyPointMatchParameters,
    scale_down_width: Option<f32>,
) -> Result<(i32, Mat), StackerError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    let files = files.into_iter().map(|p| p.as_ref().to_path_buf());
    if let Some(scale_down_width) = scale_down_width {
        keypoint_match_scale_down(files, params, scale_down_width)
    } else {
        keypoint_match_no_scale(files, params)
    }
}

fn keypoint_match_no_scale<I>(
    files: I,
    params: KeyPointMatchParameters,
) -> Result<(i32, Mat), StackerError>
where
    I: IntoIterator<Item = PathBuf>,
{
    let files_vec: Vec<PathBuf> = files.into_iter().collect();

    if files_vec.is_empty() {
        return Err(StackerError::NotEnoughFiles);
    }

    // Process the first file to get reference keypoints and descriptors
    let (first_grey_img, first_img_f32_wr) = {
        let (g, f) = utils::read_grey_and_f32(&files_vec[0], imgcodecs::IMREAD_UNCHANGED)?;
        (g, utils::UnsafeMatSyncWrapper(f))
    };

    // Get the image size from the first image for later use
    let img_size = first_grey_img.size()?;

    // Get keypoints and descriptors
    let (first_kp_wr, first_des_wr) = {
        let (kp, des) = utils::orb_detect_and_compute(&first_grey_img)?;
        (
            utils::UnsafeVectorKeyPointSyncWrapper(kp),
            utils::UnsafeMatSyncWrapper(des),
        )
    };

    // we do not need first_img anymore
    drop(first_grey_img);

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
                move |acc: Option<(i32, utils::UnsafeMatSyncWrapper)>, index| {
                    let warped_img = if index == 0 {
                        // For the first image, just use the already processed image
                        Some(first_img_f32_wrmv.0.clone())
                    } else {
                        // Process non-first images
                        // Process the first file to get reference keypoints and descriptors
                        let (grey_img, img_f32) = utils::read_grey_and_f32(&files_vec_mw[index], imgcodecs::IMREAD_UNCHANGED)?;

                        //python: src_pts = np.float32([first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        // Detect keypoints and compute descriptors
                        let (kp, des) = utils::orb_detect_and_compute(&grey_img)?;

                        // Match keypoints
                        let matches: Vec<core::DMatch> = {
                            let mut matcher = features2d::BFMatcher::create(core::NORM_HAMMING, false)?; // false for knn_match
                            matcher.add(&des)?;
                            let mut knn_matches = core::Vector::<core::Vector<core::DMatch>>::new();

                            // Explicitly calling knn_match with the required parameters
                            matcher.knn_match(
                                &first_des_wrmv.0, // query descriptors
                                &mut knn_matches,  // output matches
                                2,              // k (2 best matches per descriptor)
                                &Mat::default(),   // mask (no filtering here)
                                false,// compact_result
                            )?;

                            let mut filtered_matches = knn_matches
                                .iter()
                                .filter_map(|m| {
                                    if m.len() == 2 && m.get(0).unwrap().distance < params.match_ratio * m.get(1).unwrap().distance {
                                        Some(m.get(0).unwrap())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>();

                            // Sort by distance
                            filtered_matches.sort_by(|a, b| OrderedFloat(a.distance).cmp(&OrderedFloat(b.distance)));

                            // Keep only the best 50% of matches (optional)
                            let num_to_keep = (filtered_matches.len() as f32 * params.match_keep_ratio).round() as usize;
                            filtered_matches.truncate(num_to_keep);

                            filtered_matches
                        };
                        if matches.len() < 5 {
                            return Ok(None)
                        }

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
                        let h = match calib3d::find_homography(
                            &dst_pts,
                            &src_pts,
                            &mut Mat::default(),
                            params.method,
                            params.ransac_reproj_threshold,
                        ) {
                            Ok(matrix) => matrix,
                            Err(_) => return Ok(None), // Skip this image if homography fails
                        };

                        // Check if homography is valid
                        if h.empty() || h.rows() != 3 || h.cols() != 3 {
                            // Homography matrix is invalid
                            return Ok(None); // This means this image will be skipped
                        }

                        if core::determinant(&h)?.abs() < 1e-6 {
                            // Homography is degenerate
                            return Ok(None);
                        }

                        // Warp image
                        let mut warped_image = Mat::default();
                        imgproc::warp_perspective(
                            &img_f32,
                            &mut warped_image,
                            &h,
                            img_size,
                            imgproc::INTER_LINEAR,
                            params.border_mode,
                            params.border_value,
                        )?;

                        Some(warped_image)
                    };

                    //python: dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    // Add to the accumulator (if accumulator is empty, just use the processed image)
                    let result = match (acc, warped_img) {
                        (None, None) => (1_i32,utils::UnsafeMatSyncWrapper(first_img_f32_wrmv.0.clone())),
                        (Some((acc_d, acc_mat)), None) => (1_i32 + acc_d, acc_mat),
                        (None, Some(warped_image)) => (0_i32, utils::UnsafeMatSyncWrapper(warped_image)),
                        (Some((acc_d, acc_mat)), Some(warped_image)) => {
                            let mat = utils::UnsafeMatSyncWrapper(
                                (&acc_mat.0 + &warped_image).into_result()?.to_mat()?,
                            );
                            (acc_d, mat)
                        }
                    };

                    Ok::<Option<(i32, utils::UnsafeMatSyncWrapper)>, StackerError>(Some(result))
                },
            )
            .try_reduce(
                || None,
                |acc1, acc2| match (acc1, acc2) {
                    (None, None) => Err(StackerError::InvalidParams("All images discarded: try modifying KeyPointMatchParameters::match_distance_threshold".to_string())),
                    (Some(acc1), None) => Ok(Some(acc1)),
                    (None, Some(acc2)) => Ok(Some(acc2)),
                    (Some(acc1), Some(acc2)) => {
                        // Combine the two accumulators
                        let combined_img = utils::UnsafeMatSyncWrapper(
                            (&acc1.1.0 + &acc2.1.0).into_result()?.to_mat()?,
                        );
                        Ok(Some((acc1.0+ acc2.0, combined_img)))
                    }
                },
            )
    }?;

    // Final normalization
    let final_result = if let Some((dropped, img)) = result {
        (
            dropped,
            (img.0 / (files_vec.len() - dropped as usize) as f64)
                .into_result()?
                .to_mat()?,
        )
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
    scale_down_width: f32,
) -> Result<(i32, Mat), StackerError>
where
    I: IntoIterator<Item = PathBuf>,
{
    let files_vec: Vec<PathBuf> = files.into_iter().collect();
    if files_vec.is_empty() {
        return Err(StackerError::NotEnoughFiles);
    }

    // Process the first file to get reference keypoints and descriptors
    let (grey_first_img, first_img_f32_wr) = {
        let (g, f) = utils::read_grey_and_f32(&files_vec[0], imgcodecs::IMREAD_UNCHANGED)?;
        (g, utils::UnsafeMatSyncWrapper(f))
    };

    // Get the image size from the first image for later use
    let full_size = first_img_f32_wr.0.size()?;

    if scale_down_width >= full_size.width as f32 {
        return Err(StackerError::InvalidParams(format!(
            "scale_down_to was larger (or equal) to the full image width: full_size:{}, scale_down_to:{}",
            full_size.width, scale_down_width
        )));
    }

    // Get keypoints and descriptors
    let (first_kp_small_wr, first_des_small_wr) = {
        // Scale down the first image
        let grey_img_small = utils::scale_image(&grey_first_img, scale_down_width)?;

        let (kp, des) = utils::orb_detect_and_compute(&grey_img_small)?;
        (
            utils::UnsafeVectorKeyPointSyncWrapper(kp),
            utils::UnsafeMatSyncWrapper(des),
        )
    };

    // we do not need grey_first_img anymore
    drop(grey_first_img);

    // Combined map-reduce step: Process each image and reduce on the fly
    let result = {
        // Create a reference to our wrappers that we can move into the closure
        let first_kp_small_wrmv = &first_kp_small_wr;
        let first_des_small_wrmv = &first_des_small_wr;
        let first_img_f32_wrmv = &first_img_f32_wr;
        let files_vec_wr = &files_vec;

        (0..files_vec.len())
            .into_par_iter()
            //.into_iter()
            .try_fold(
                || None, // Initial accumulator for each thread
                move |acc: Option<(i32, utils::UnsafeMatSyncWrapper)>, index| {
                    let warped_img = if index == 0 {
                        // For the first image, just use the already processed image
                        Some(first_img_f32_wrmv.0.clone())
                    } else {
                        // Process non-first images

                        // Load and scale down the image for feature detection
                        let (img_f32, grey_img_small) = {
                            let (grey_img, img_f32) = utils::read_grey_and_f32(
                                &files_vec_wr[index],
                                imgcodecs::IMREAD_UNCHANGED,
                            )?;
                            let grey_img_small = utils::scale_image(&grey_img, scale_down_width)?;
                            (img_f32, grey_img_small)
                        };

                        // Detect features on the scaled-down image
                        let (kp_small, des_small) = utils::orb_detect_and_compute(&grey_img_small)?;

                        // Match features
                        // Match keypoints
                        let matches: Vec<core::DMatch> = {
                            let mut matcher =
                                features2d::BFMatcher::create(core::NORM_HAMMING, false)?; // false for knn_match
                            matcher.add(&des_small)?;
                            let mut knn_matches = core::Vector::<core::Vector<core::DMatch>>::new();

                            // Explicitly calling knn_match with the required parameters
                            matcher.knn_match(
                                &first_des_small_wrmv.0, // query descriptors
                                &mut knn_matches,        // output matches
                                2,                       // k (2 best matches per descriptor)
                                &Mat::default(),         // mask (no filtering here)
                                false,                   // compact_result
                            )?;

                            let mut filtered_matches = knn_matches
                                .iter()
                                .filter_map(|m| {
                                    if m.len() == 2
                                        && m.get(0).unwrap().distance
                                            < params.match_ratio * m.get(1).unwrap().distance
                                    {
                                        Some(m.get(0).unwrap())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>();

                            // Sort by distance
                            filtered_matches.sort_by(|a, b| {
                                OrderedFloat(a.distance).cmp(&OrderedFloat(b.distance))
                            });

                            // Keep only the best 50% of matches (optional)
                            let num_to_keep = (filtered_matches.len() as f32
                                * params.match_keep_ratio)
                                .round() as usize;
                            filtered_matches.truncate(num_to_keep);

                            filtered_matches
                        };
                        if matches.len() < 5 {
                            return Ok(None);
                        }

                        // Get corresponding points (on small images)
                        let src_pts = {
                            let mut pts: Vector<Point2f> = Vector::with_capacity(matches.len());
                            for m in matches.iter() {
                                pts.push(first_kp_small_wrmv.0.get(m.query_idx as usize)?.pt());
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
                        let h_small = match calib3d::find_homography(
                            &dst_pts,
                            &src_pts,
                            &mut Mat::default(),
                            params.method,
                            params.ransac_reproj_threshold,
                        ) {
                            Ok(matrix) => matrix,
                            Err(_) => return Ok(None), // Skip this image if homography fails
                        };

                        // Check if homography is valid
                        if h_small.empty() || h_small.rows() != 3 || h_small.cols() != 3 {
                            // Homography matrix is invalid
                            return Ok(None); // This means this image will be skipped
                        }

                        if core::determinant(&h_small)?.abs() < 1e-6 {
                            // Homography is degenerate
                            return Ok(None);
                        }

                        // Scale homography matrix to apply to original-sized images
                        let h = utils::adjust_homography_for_scale_f64(
                            &h_small,
                            &grey_img_small,
                            &img_f32,
                        )?;

                        // Warp the original full-sized image
                        let mut warped_image = Mat::default();

                        imgproc::warp_perspective(
                            &img_f32,
                            &mut warped_image,
                            &h,
                            full_size,
                            imgproc::INTER_LINEAR,
                            params.border_mode,
                            params.border_value,
                        )?;

                        Some(warped_image)
                    };
                    // Add to the accumulator (if accumulator is empty, just use the processed image)
                    let result = match (acc, warped_img) {
                        (None, None) => (
                            1_i32,
                            utils::UnsafeMatSyncWrapper(first_img_f32_wrmv.0.clone()),
                        ),
                        (Some((acc_d, acc_mat)), None) => (1_i32 + acc_d, acc_mat),
                        (None, Some(warped_image)) => {
                            (0_i32, utils::UnsafeMatSyncWrapper(warped_image))
                        }
                        (Some((acc_d, acc_mat)), Some(warped_image)) => {
                            let mat = utils::UnsafeMatSyncWrapper(
                                (&acc_mat.0 + &warped_image).into_result()?.to_mat()?,
                            );
                            (acc_d, mat)
                        }
                    };

                    Ok::<Option<(i32, utils::UnsafeMatSyncWrapper)>, StackerError>(Some(result))
                },
            )
            .try_reduce(
                || None,
                |acc1, acc2| match (acc1, acc2) {
                    (Some(acc1), None) => Ok(Some(acc1)),
                    (None, Some(acc2)) => Ok(Some(acc2)),
                    (Some(acc1), Some(acc2)) => {
                        // Combine the two accumulators
                        let combined_img = utils::UnsafeMatSyncWrapper(
                            (&acc1.1.0 + &acc2.1.0).into_result()?.to_mat()?,
                        );
                        Ok(Some((acc1.0 + acc2.0, combined_img)))
                    }
                    _ => unreachable!(),
                },
            )
    }?;

    // Final normalization
    let final_result = if let Some((dropped, img)) = result {
        (
            dropped,
            (img.0 / (files_vec.len() - dropped as usize) as f64)
                .into_result()?
                .to_mat()?,
        )
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

/// Aligns and stacks images using OpenCV's Enhanced Correlation Coefficient (ECC) algorithm.
///
/// Performs intensity-based image alignment using pyramidal ECC maximization, with optional
/// scaling for performance. The first image serves as the reference frame for alignment.
///
/// # Algorithm Overview
/// 1. Converts images to grayscale for alignment computation
/// 2. Optionally scales images down for faster ECC computation
/// 3. Estimates transformation (translation/affine/homography) using ECC maximization
/// 4. Applies transformation to original full-resolution images
/// 5. Averages all aligned images
///
/// # When to Use
/// - When images need alignment but lack distinctive features for keypoint matching
/// - For alignment under illumination changes (ECC is illumination-invariant)
/// - When precise subpixel alignment is required
///
/// # Parameters
/// - `files`: Iterator of image paths (any `AsRef<Path>` type)
/// - `params`: ECC configuration parameters:
///   - `motion_type`: Type of transformation to estimate:
///     - `Translation`, `Euclidean`, `Affine`, or `Homography`
///   - `max_count`: Maximum iterations (None for OpenCV default)
///   - `epsilon`: Convergence threshold (None for OpenCV default)
///   - `gauss_filt_size`: Gaussian filter size for pyramid levels (odd number ≥1)
/// - `scale_down_width`: Performance/accuracy trade-off:
///   - `Some(width)`: Process ECC at specified width (faster)
///   - `None`: Process at full resolution (more precise)
///   - Must be < original width when specified
///
/// # Returns
/// - `Ok(Mat)`: Stacked image in CV_32F format (normalized 0-1 range)
/// - `Err(StackerError)` on:
///   - No input files
///   - Invalid scale width
///   - Image loading/processing failures
///
/// # Performance Notes
/// - Parallel processing for multi-image alignment
/// - Scaling provides 3-5x speedup with minor accuracy impact
/// - Computation complexity depends on:
///   - Motion type (Translation < Affine < Homography)
///   - Image resolution
///   - Convergence criteria
///
/// # Example
/// ```rust,no_run
/// # use libstacker::{prelude::*, opencv::prelude::*};
/// # fn main() -> Result<(), StackerError> {
/// // Fast homography alignment with scaling
/// let aligned = ecc_match(
///     ["1.jpg", "2.jpg", "3.jpg"],
///     EccMatchParameters {
///         motion_type: MotionType::Homography,
///         max_count: Some(5000),
///         epsilon: Some(1e-6),
///         gauss_filt_size: 3,
///     },
///     Some(800.0)  // Scale to 800px width
/// )?;
///
/// // Precise affine alignment (full resolution)
/// let precise = ecc_match(
///     vec!["1.tif", "2.tif"],
///     EccMatchParameters {
///         motion_type: MotionType::Affine,
///         max_count: Some(5000),
///         epsilon: Some(1e-6),
///         gauss_filt_size: 3,
///     },
///     None  // No scaling
/// )?;
/// # Ok(())}
/// ```
///
/// # References
/// - [Image Alignment Tutorial](https://learnopencv.com/image-alignment-ecc-in-opencv-c-python)
pub fn ecc_match<I, P>(
    files: I,
    params: EccMatchParameters,
    scale_down_width: Option<f32>,
) -> Result<Mat, StackerError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    let files = files.into_iter().map(|p| p.as_ref().to_path_buf());
    if let Some(scale_down_width) = scale_down_width {
        ecc_match_scaling_down(files, params, scale_down_width)
    } else {
        ecc_match_no_scaling(files, params)
    }
}

fn ecc_match_no_scaling<I>(files: I, params: EccMatchParameters) -> Result<Mat, StackerError>
where
    I: IntoIterator<Item = PathBuf>,
{
    let files_vec: Vec<PathBuf> = files.into_iter().collect();

    if files_vec.is_empty() {
        return Err(StackerError::NotEnoughFiles);
    }

    let criteria = Result::<core::TermCriteria, StackerError>::from(params)?;

    let (first_img_grey_wr, first_img_f32_wr) = {
        let (img_grey, first_img_f32) =
            utils::read_grey_and_f32(&files_vec[0], imgcodecs::IMREAD_UNCHANGED)?;
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
            .with_min_len(1)
            .try_fold(
                || None, // Initial accumulator for each thread
                move |acc: Option<utils::UnsafeMatSyncWrapper>, index| {
                    let warped_image = if index == 0 {
                        // For the first image, just use the already processed image
                        first_img_f32_wrmv.0.clone()
                    } else {
                        let (img_grey, img_f32) = utils::read_grey_and_f32(
                            &files_vec_mv[index],
                            imgcodecs::IMREAD_UNCHANGED,
                        )?;

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

fn ecc_match_scaling_down<I>(
    files: I,
    params: EccMatchParameters,
    scale_down_width: f32,
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
        let (img_grey, first_img_f32) =
            utils::read_grey_and_f32(&files_vec[0], imgcodecs::IMREAD_UNCHANGED)?;
        (
            utils::UnsafeMatSyncWrapper(img_grey),
            utils::UnsafeMatSyncWrapper(first_img_f32),
        )
    };

    // Load first image in color and grayscale
    let full_size = first_img_f32_wr.0.size()?;
    if scale_down_width >= full_size.width as f32 {
        return Err(StackerError::InvalidParams(format!(
            "scale_down_to was larger (or equal) to the full image width: full_size:{}, scale_down_to:{}",
            full_size.width, scale_down_width
        )));
    }

    if scale_down_width <= 10.0 {
        return Err(StackerError::InvalidParams(format!(
            "scale_down_to was too small scale_down_to:{}",
            scale_down_width
        )));
    }

    // Scale down the first image for ECC
    let first_img_grey_small_wr =
        utils::UnsafeMatSyncWrapper(utils::scale_image(&first_img_grey_wr.0, scale_down_width)?);
    let small_size = first_img_grey_small_wr.0.size()?;

    // Combined map-reduce step: Process each image and reduce on the fly
    let result = {
        let first_img_f32_wrmv = &first_img_f32_wr;
        let files_vec_mv = &files_vec;

        (0..files_vec.len())
            .into_par_iter()
            .with_min_len(1)
            .try_fold(
                || None, // Initial accumulator for each thread
                move |acc: Option<utils::UnsafeMatSyncWrapper>, index| {
                    let warped_image = if index == 0 {
                        // For the first image, just use the already processed image
                        first_img_f32_wrmv.0.clone()
                    } else {
                        // Load image in both color and grayscale
                        let (img_grey_original, img_f32) = utils::read_grey_and_f32(
                            &files_vec_mv[index],
                            imgcodecs::IMREAD_UNCHANGED,
                        )?;
                        let first_img_grey_small = &first_img_grey_small_wr;

                        // Scale down for ECC matching
                        let img_small = utils::scale_image(&img_f32, scale_down_width)?;
                        let img_grey_small =
                            utils::scale_image(&img_grey_original, scale_down_width)?;

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
