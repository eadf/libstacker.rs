[![crates.io](https://img.shields.io/crates/v/libstacker.svg)](https://crates.io/crates/libstacker)
[![Documentation](https://docs.rs/libstacker/badge.svg)](https://docs.rs/libstacker)
[![Workflow](https://github.com/eadf/libstacker.rs/workflows/Rust/badge.svg)](https://github.com/eadf/libstacker.rs/workflows/Rust/badge.svg)
[![dependency status](https://deps.rs/crate/libstacker/0.0.7/status.svg)](https://deps.rs/crate/libstacker/0.0.7)
![license](https://img.shields.io/crates/l/libstacker)

# libstacker
A multithreaded port of the python code found [here: github.com/maitek/image_stacking](https://github.com/maitek/image_stacking) 

This crate contains multithreaded functions that aligns and stacks images using [OpenCV](https://crates.io/crates/opencv) and [Rayon](https://crates.io/crates/rayon).

Read more about image alignment with OpenCV [here](https://learnopencv.com/image-alignment-ecc-in-opencv-c-python).

## Usage:
### Download the test images:

```git clone https://github.com/maitek/image_stacking image_stacking_py```

### Build the code:
Opencv-rust can be little tricky to install. Follow the instructions from [rust opencv](https://crates.io/crates/opencv)

You will need the "clang-runtime" feature if you <a href="https://github.com/twistedfall/opencv-rust#Troubleshooting">experience problems with your clang environment</a>
.

```bash
cargo build --release
```

or

```bash
cargo build --release --features "clang-runtime"
```

### Run the example:

```bash
cargo run --example main --release
```

or

```bash
cargo run --example main --release --features "clang-runtime"
```

and then wait a few seconds. The program should sort the images by quality, drop the least sharp image, and align and stack the rest. 
The result should be two windows showing the stacked images using two different alignment methods.

## API
```rust
let keypoint_match_img:opencv::core::Mat = keypoint_match(
   // a Vec<PathBuf> containing paths to image files
   collect_image_files(&PathBuf::from("image_stacking_py/images"))?,
   KeyPointMatchParameters {
      method: opencv::calib3d::RANSAC,
      ransac_reproj_threshold: 5.0,
   },
)?;
```

Depending on the parameters the `ecc_match()` is much slower, but also more accurate. 
```rust
let ecc_match_img:opencv::core::Mat = ecc_match(
   // a Vec<PathBuf> containing paths to image files
   collect_image_files(&PathBuf::from("image_stacking_py/images"))?,
   EccMatchParameters {
      motion_type: MotionType::Homography,
      max_count: Some(5000),
      epsilon: Some(1e-5),
      gauss_filt_size: 5,
   },
)?;
```

## Todo

* Figure out the docs.rs problem
* Figure out what to do with `.reshape()` in `keypoint_match()`
* ~~Figure out some opencv parameters~~ responsibility sneakily shifted to end user.
* Complete the `sharpness_tenengrad()` function. Mat not square?
* Command line options in the example

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0)>
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT)>

at your option.
