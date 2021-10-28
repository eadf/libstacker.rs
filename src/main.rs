use libstacker::{ecc_match, keypoint_match, StackerError};
use opencv::highgui::{imshow, wait_key};
use std::path::PathBuf;

fn main() -> Result<(), StackerError> {
    let now = std::time::Instant::now();
    let keypoint_match_img = keypoint_match(PathBuf::from("image_stacking_py/images"))?;
    println!("Calculated keypoint_match() in {:?}", now.elapsed());

    let now = std::time::Instant::now();
    let ecc_match_img = ecc_match(PathBuf::from("image_stacking_py/images"))?;
    println!("Calculated ecc_match() in {:?}", now.elapsed());

    while wait_key(33)? != 27 {
        let _ = imshow("KeyPoint match", &keypoint_match_img)?;
        let _ = imshow("ECC match", &ecc_match_img)?;
    }
    Ok(())
}
