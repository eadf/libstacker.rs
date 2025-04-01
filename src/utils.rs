use super::{EccMatchParameters, StackerError};
use opencv::core::{AlgorithmHint, Mat, MatTrait, MatTraitConst};
use opencv::features2d::ORB;
use opencv::prelude::Feature2DTrait;
use opencv::{imgcodecs, imgproc};
use std::path;

/// Extension trait for more ergonomic Mat conversions
pub trait MatExt {
    /// Convert matrix to specified type with default scaling
    ///
    /// # Arguments
    /// * `target_type` - OpenCV type constant (e.g., CV_32F, CV_64F)
    fn convert(&self, rtype: i32, alpha: f64, beta: f64) -> Result<Mat, StackerError>;
}

impl MatExt for Mat {
    fn convert(&self, rtype: i32, alpha: f64, beta: f64) -> Result<Mat, StackerError> {
        let mut dst = Mat::default();
        self.convert_to(&mut dst, rtype, alpha, beta)?;
        Ok(dst)
    }
}

/// A q&d hack allowing `opencv::Vector<KeyPoints>` objects to be `Sync`.
/// Only use this on immutable `Vector<KeyPoints>` objects.
pub(super) struct UnsafeVectorKeyPointSyncWrapper(
    pub(super) opencv::core::Vector<opencv::core::KeyPoint>,
);
unsafe impl Sync for UnsafeVectorKeyPointSyncWrapper {}

/// A q&d hack allowing `opencv::Mat` objects to be `Sync`.
/// Only use this on immutable `Mat` objects.
pub(super) struct UnsafeMatSyncWrapper(pub(super) Mat);
unsafe impl Sync for UnsafeMatSyncWrapper {}

/// Trait for setting value in a 2d Mat<T>
/// Todo:There must be a better way to do this
pub trait SetMValue {
    fn set_2d<T: opencv::prelude::DataType>(
        &mut self,
        row: i32,
        col: i32,
        value: T,
    ) -> Result<(), StackerError>;
}

impl SetMValue for Mat {
    #[inline]
    /// ```
    /// # use libstacker::{prelude::*, opencv::prelude::* ,opencv::prelude::MatTraitConst};
    /// # use crate::libstacker::utils::SetMValue;
    /// let mut m = unsafe { opencv::core::Mat::new_rows_cols(1, 3, opencv::core::CV_64FC1).unwrap() };
    /// m.set_2d::<f64>(0, 0, -1.0).unwrap();
    /// m.set_2d::<f64>(0, 1, -2.0).unwrap();
    /// m.set_2d::<f64>(0, 2, -3.0).unwrap();
    /// assert_eq!(-1.0, *m.at_2d::<f64>(0,0).unwrap());
    /// assert_eq!(-2.0, *m.at_2d::<f64>(0,1).unwrap());
    /// assert_eq!(-3.0, *m.at_2d::<f64>(0,2).unwrap());
    /// ```
    fn set_2d<T: opencv::prelude::DataType>(
        &mut self,
        row: i32,
        col: i32,
        value: T,
    ) -> Result<(), StackerError> {
        let v = self.at_2d_mut::<T>(row, col)?;
        *v = value;
        Ok(())
    }
}

/// Safe wrapper around OpenCV's `imread` with proper error handling
///
/// This exists because:
/// 1. OpenCV's API requires a `&str` path rather than standard Rust `Path` types
/// 2. Paths might contain non-Unicode characters that need proper error handling
///
/// # Features
/// - Converts `Path`-like inputs to strings with validation
/// - Returns domain-specific errors (`StackerError`) instead of raw OpenCV errors
///
/// # Arguments
/// * `path` - Filesystem path to image (any type implementing `AsRef<Path>`)
/// * `flags` - OpenCV loading flags (e.g., `imgcodecs::IMREAD_COLOR`)
///
/// # Errors
/// Returns `StackerError::InvalidPathEncoding` if:
/// - Path contains invalid Unicode characters
///
/// Returns `StackerError::OpenCvError` with metadata if:
/// - File not found
/// - Unsupported format
/// - Insufficient permissions
/// - Corrupted image data
/// - Invalid OpenCV flags
///
/// # Example
/// ```no_run
/// # use libstacker::{utils::imread, prelude::*, opencv::prelude::*, opencv::imgcodecs };
/// # use std::path::Path;
/// # fn a() -> Result<(),StackerError> {
/// let img = imread("image.jpg", imgcodecs::IMREAD_GRAYSCALE)?;
/// match imread(Path::new("image.png"), imgcodecs::IMREAD_COLOR) {
///     Ok(mat) => /* process image */(),
///     Err(_) => /* handle opencv error */(),
/// }
/// # Ok(()) }
/// ```
#[inline(always)]
pub fn imread<P: AsRef<std::path::Path>>(path: P, flags: i32) -> Result<Mat, StackerError> {
    let path_str = path
        .as_ref()
        .to_str()
        .ok_or_else(|| StackerError::InvalidPathEncoding(path.as_ref().to_path_buf()))?;
    Ok(imgcodecs::imread(path_str, flags)?)
}

/// Read a image file and returns a (`Mat<COLOR_BGR2GRAY>`,`Mat<CV_32F>`) tuple
pub(super) fn read_grey_f32(filename: &path::Path) -> Result<(Mat, Mat), StackerError> {
    let img = imgcodecs::imread(filename.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
    let img_f32 = img.convert(opencv::core::CV_32F, 1.0 / 255.0, 0.0)?;

    let mut img_grey = Mat::default();
    imgproc::cvt_color(
        &img,
        &mut img_grey,
        imgproc::COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok((img_grey, img_f32))
}

impl From<EccMatchParameters> for Result<opencv::core::TermCriteria, StackerError> {
    /// Converts from a `EccMatchParameters` to TermCriteria
    /// ```
    /// # use libstacker::{prelude::*, opencv::prelude::*, opencv::core::TermCriteria_Type};
    /// let t:Result<opencv::core::TermCriteria, StackerError> = EccMatchParameters{
    ///     motion_type: MotionType::Euclidean,
    ///     max_count: None,
    ///     epsilon: Some(0.1),
    ///     gauss_filt_size:3}.into();
    /// let t = t.unwrap();
    /// assert_eq!(t.epsilon, 0.1);
    /// assert_eq!(t.typ, TermCriteria_Type::EPS as i32);
    /// ```
    fn from(r: EccMatchParameters) -> Result<opencv::core::TermCriteria, StackerError> {
        let mut rv = opencv::core::TermCriteria::default()?;
        if let Some(max_count) = r.max_count {
            rv.typ |= opencv::core::TermCriteria_Type::COUNT as i32;
            rv.max_count = max_count;
        }
        if let Some(epsilon) = r.epsilon {
            rv.typ |= opencv::core::TermCriteria_Type::EPS as i32;
            rv.epsilon = epsilon;
        }
        Ok(rv)
    }
}

/// Does a (`Mat<f32>`, key-points and descriptors) tuple of a Mat
pub(super) fn orb_detect_and_compute(
    img: &Mat,
) -> Result<(opencv::core::Vector<opencv::core::KeyPoint>, Mat), StackerError> {
    let mut orb = ORB::create_def()?;

    let mut kp = opencv::core::Vector::<opencv::core::KeyPoint>::new();
    let mut des = Mat::default();
    orb.detect_and_compute(img, &Mat::default(), &mut kp, &mut des, false)?;
    Ok((kp, des))
}

// Helper function to scale an image while maintaining aspect ratio
pub(super) fn scale_image(img: &Mat, scale_down: f32) -> Result<Mat, StackerError> {
    let size = img.size()?;
    let width = size.width;
    let height = size.height;

    // Calculate scaling factor to make the smaller dimension equal to scale_down
    let scaling_factor = if width < height {
        scale_down as f64 / width as f64
    } else {
        scale_down as f64 / height as f64
    };

    // Calculate new dimensions
    let new_width = (width as f64 * scaling_factor) as i32;
    let new_height = (height as f64 * scaling_factor) as i32;

    // Resize the image
    let mut resized = Mat::default();
    imgproc::resize(
        img,
        &mut resized,
        opencv::core::Size::new(new_width, new_height),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;

    Ok(resized)
}

// Helper function to adjust homography matrix for the scale difference
// Define a macro to create adjust_homography_for_scale implementations for different types
macro_rules! impl_adjust_homography_for_scale {
    ($name:ident, $type:ty) => {
        pub(super) fn $name(
            h_small: &Mat,
            img_small: &Mat,
            img_original: &Mat,
        ) -> Result<Mat, StackerError> {
            let small_size = img_small.size()?;
            let orig_size = img_original.size()?;

            // Calculate scaling factors
            let scale_x = orig_size.width as f64 / small_size.width as f64;
            let scale_y = orig_size.height as f64 / small_size.height as f64;

            // Create a copy of the homography matrix
            let mut h_adjusted = h_small.clone();

            // Adjust the homography matrix elements directly
            *h_adjusted.at_2d_mut::<$type>(0, 2)? *= scale_x as $type;
            *h_adjusted.at_2d_mut::<$type>(1, 2)? *= scale_y as $type;
            *h_adjusted.at_2d_mut::<$type>(2, 0)? /= scale_x as $type;
            *h_adjusted.at_2d_mut::<$type>(2, 1)? /= scale_y as $type;

            Ok(h_adjusted)
        }
    };
}

// Implement for both f32 and f64
impl_adjust_homography_for_scale!(adjust_homography_for_scale_f64, f64);
impl_adjust_homography_for_scale!(adjust_homography_for_scale_f32, f32);
