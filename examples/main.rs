// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2021,2025 lacklustr@protonmail.com https://github.com/eadf

use libstacker::opencv::{highgui, imgcodecs};
use libstacker::prelude::*;
use rayon::prelude::*;
use std::path;

/// Returns paths to all jpg,jpeg,tif and png files in a single directory (non-recursive)
pub fn collect_image_files(
    path: &std::path::Path,
) -> Result<Vec<std::path::PathBuf>, StackerError> {
    Ok(std::fs::read_dir(path)?
        .flatten()
        .filter_map(|f| f.path().is_file().then(|| f.path()))
        .filter(|p| p.extension().is_some() && p.extension().unwrap().to_str().is_some())
        .filter(|p| {
            let extension = p.extension().unwrap().to_str().unwrap().to_uppercase();
            extension == "JPG" || extension == "JPEG" || extension == "TIF" || extension == "PNG"
        })
        .collect())
}

/// Run an example aligning images found under "image_stacking_py/images"
/// The example displays two images, one is aligned with the `keypoint_match()` method,
/// the other with `ecc_match()`
fn main() -> Result<(), StackerError> {
    use libstacker::opencv::core::Point;

    let v = vec![Point::new(1, 5), Point::new(2, 5), Point::new(3, 5)];
    let _ = v
        .into_iter()
        .collect::<libstacker::opencv::core::Vector<Point>>();

    let files = collect_image_files(&path::PathBuf::from("image_stacking_py/images"))?;
    let now = std::time::Instant::now();
    let mut files = files
        .into_par_iter()
        .map(move |f| -> Result<_, StackerError> {
            let img_gray = imgcodecs::imread(f.to_str().unwrap(), imgcodecs::IMREAD_GRAYSCALE)?;
            Ok((
                f,
                libstacker::sharpness_modified_laplacian(&img_gray)?, // LAPM = 1
                libstacker::sharpness_variance_of_laplacian(&img_gray)?, // LAPV = 2
                libstacker::sharpness_tenengrad(&img_gray, 3)?,       // TENG = 3
                libstacker::sharpness_normalized_gray_level_variance(&img_gray)?, // GLVN = 4
            ))
        })
        .collect::<Result<Vec<_>, StackerError>>()?;
    println!("Calculated sharpness() in {:?}", now.elapsed());

    // sort images by sharpness using TENG for now
    files.sort_by_key(|f| ordered_float::OrderedFloat(f.3));

    println!("Files ordered by TENG (low quality first)");
    for f in files.iter() {
        println!(
            "{:?} LAPM:{: >8.5} LAPV:{: >9.5} TENG:{: >8.5} GLVN:{: >9.5}",
            f.0, f.1, f.2, f.3, f.4
        );
    }
    // only keep the filename and skip the first file (the bad one) and reverse the order
    // keeping the best image first.
    let files: Vec<_> = files.into_iter().map(|f| f.0).skip(1).rev().collect();

    let now = std::time::Instant::now();
    let keypoint_match_img = keypoint_match(
        &files,
        libstacker::KeyPointMatchParameters {
            method: opencv::calib3d::RANSAC,
            ransac_reproj_threshold: 5.0,
        },
        None,
        //Some(720.0),
    )?;
    println!("Calculated keypoint_match() in {:?}", now.elapsed());

    let now = std::time::Instant::now();
    let keypoint_match_img_400 = keypoint_match(
        &files,
        libstacker::KeyPointMatchParameters {
            method: opencv::calib3d::RANSAC,
            ransac_reproj_threshold: 5.0,
        },
        Some(400.0),
    )?;
    println!("Calculated keypoint_match(size=400) in {:?}", now.elapsed());

    let now = std::time::Instant::now();
    let ecc_match_img = libstacker::ecc_match(
        &files,
        libstacker::EccMatchParameters {
            motion_type: libstacker::MotionType::Homography,
            max_count: Some(5000),
            epsilon: Some(1e-5),
            gauss_filt_size: 5,
        },
        None,
    )?;
    println!("Calculated ecc_match() in {:?}", now.elapsed());

    let now = std::time::Instant::now();
    let ecc_match_img_400 = libstacker::ecc_match(
        files,
        libstacker::EccMatchParameters {
            motion_type: libstacker::MotionType::Homography,
            max_count: Some(5000),
            epsilon: Some(1e-5),
            gauss_filt_size: 5,
        },
        Some(400.0),
    )?;
    println!("Calculated ecc_match(size=400) in {:?}", now.elapsed());

    while highgui::wait_key(33)? != 27 {
        highgui::imshow("KeyPoint match", &keypoint_match_img)?;
        highgui::imshow("ECC match", &ecc_match_img)?;
        highgui::imshow("KeyPoint match size 400", &keypoint_match_img_400)?;
        highgui::imshow("ECC match size 400", &ecc_match_img_400)?;
    }
    Ok(())
}
