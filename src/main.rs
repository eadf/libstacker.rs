use libstacker::{keypoint_match, StackerError, ecc_match};
use opencv::highgui::{imshow, wait_key};
use std::path::PathBuf;

fn main() -> Result<(), StackerError> {
    let keypoint_match_img = keypoint_match(PathBuf::from("image_stacking_py/images"))?;
    let ecc_match_img = ecc_match(PathBuf::from("image_stacking_py/images"))?;
    while wait_key(33)? != 27 {
        let _ = imshow("KeyPoint match", &keypoint_match_img)?;
        let _ = imshow("ECC match", &ecc_match_img)?;
    }
    Ok(())
}
