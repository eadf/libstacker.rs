# libstacker.rs
A port of the python code found [here: github.com/maitek/image_stacking](https://github.com/maitek/image_stacking) 

## Usage:
### Download the test images:

```git clone https://github.com/maitek/image_stacking image_stacking_py```

### Build the code:
opencv-rust can be little tricky to install. Follow instructions from here: [rust opencv](https://crates.io/crates/opencv)

```cargo build --release```

### Run the example:

```cargo run --example main --release```

and then wait a few seconds..


## Todo

* Figure out what to do with .reshape() in keypoint_match()
* Figure out some opencv parameters
* Command line options in the example
