use libstacker::opencv::{highgui, imgcodecs};
use rayon::prelude::*;
use std::path;

/// Run an example aligning images found under "image_stacking_py/images"
/// The example displays two images, one is aligned with the `keypoint_match()` method,
/// the other with `ecc_match()`
fn main() -> Result<(), libstacker::StackerError> {
    let files = libstacker::collect_image_files(&path::PathBuf::from("image_stacking_py/images"))?;
    let now = std::time::Instant::now();
    let mut files = files
        .into_par_iter()
        .map(move |f| -> Result<_, libstacker::StackerError> {
            let img_gray = imgcodecs::imread(f.to_str().unwrap(), imgcodecs::IMREAD_GRAYSCALE)?;
            Ok((
                f,
                libstacker::sharpness_modified_laplacian(&img_gray)?,
                libstacker::sharpness_variance_of_laplacian(&img_gray)?,
                libstacker::sharpness_tenengrad(&img_gray, 3)?,
                libstacker::sharpness_normalized_gray_level_variance(&img_gray)?,
            ))
        })
        .collect::<Result<Vec<_>, libstacker::StackerError>>()?;
    println!("Calculated sharpness() in {:?}", now.elapsed());

    // sort images by sharpness using LAPM for now
    files.sort_by_key(|f| ordered_float::OrderedFloat(f.1));

    for f in files.iter() {
        println!(
            "{:?} LAPM:{: >8.5} LAPV:{: >9.5} TENG:{: >8.5} GLVN:{: >9.5}",
            f.0, f.1, f.2, f.3, f.4
        );
    }
    // only keep the filename and skip the last file (the bad one)
    let files: Vec<_> = files.into_iter().map(|f| f.0).rev().skip(1).rev().collect();

    let now = std::time::Instant::now();
    let keypoint_match_img = libstacker::keypoint_match(
        files.clone(),
        libstacker::KeyPointMatchParameters {
            method: opencv::calib3d::RANSAC,
            ransac_reproj_threshold: 5.0,
        },
    )?;
    println!("Calculated keypoint_match() in {:?}", now.elapsed());

    let now = std::time::Instant::now();
    let ecc_match_img = libstacker::ecc_match(
        files,
        libstacker::EccMatchParameters {
            motion_type: libstacker::MotionType::Homography,
            max_count: Some(5000),
            epsilon: Some(1e-5),
            gauss_filt_size: 5,
        },
    )?;
    println!("Calculated ecc_match() in {:?}", now.elapsed());

    while highgui::wait_key(33)? != 27 {
        let _ = highgui::imshow("KeyPoint match", &keypoint_match_img)?;
        let _ = highgui::imshow("ECC match", &ecc_match_img)?;
    }
    Ok(())
}
