use libstacker::{
    collect_image_files, ecc_match, keypoint_match, EccMatchParameters, KeyPointMatchParameters,
    MotionType, StackerError,
};
use opencv::highgui::{imshow, wait_key};
use std::path::PathBuf;

fn main() -> Result<(), StackerError> {
    let now = std::time::Instant::now();
    let keypoint_match_img = keypoint_match(
        collect_image_files(&PathBuf::from("image_stacking_py/images"))?,
        KeyPointMatchParameters {
            method: opencv::calib3d::RANSAC,
            ransac_reproj_threshold: 5.0,
        },
    )?;
    println!("Calculated keypoint_match() in {:?}", now.elapsed());

    let now = std::time::Instant::now();
    let ecc_match_img = ecc_match(
        collect_image_files(&PathBuf::from("image_stacking_py/images"))?,
        EccMatchParameters {
            motion_type: MotionType::Homography,
            max_count: Some(5000),
            epsilon: Some(1e-5),
            gauss_filt_size: 5,
        },
    )?;
    println!("Calculated ecc_match() in {:?}", now.elapsed());

    while wait_key(33)? != 27 {
        let _ = imshow("KeyPoint match", &keypoint_match_img)?;
        let _ = imshow("ECC match", &ecc_match_img)?;
    }
    Ok(())
}
