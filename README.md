# libstacker.rs
A multi-threaded port of the python code found [here: github.com/maitek/image_stacking](https://github.com/maitek/image_stacking) 

## Usage:
### Download the test images:

```git clone https://github.com/maitek/image_stacking image_stacking_py```

### Build the code:
opencv-rust can be little tricky to install. Follow instructions from here: [rust opencv](https://crates.io/crates/opencv)

```cargo build --release```

### Run the example:

```cargo run --example main --release```

and then wait a few seconds..

## API
```rust
let keypoint_match_img = keypoint_match(
   collect_image_files(&PathBuf::from("image_stacking_py/images"))?,
   KeyPointMatchParameters {
      method: opencv::calib3d::RANSAC,
      ransac_reproj_threshold: 5.0,
   },
)?;
```

Depending on the parameters the `ecc_match()` is much slower, but also more accurate. 
```rust
let ecc_match_img = ecc_match(
   collect_image_files(&PathBuf::from("image_stacking_py/images"))?,
      EccMatchParameters {
         max_count: Some(5000),
         epsilon: Some(1e-5),
         gauss_filt_size: 5,
      },
)?;
```

## Todo

* Figure out what to do with `.reshape()` in `keypoint_match()`
* ~~Figure out some opencv parameters~~ responsibility shifted to end user.
* Command line options in the example
