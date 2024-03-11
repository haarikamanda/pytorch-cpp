// Copyright 2020-present pytorch-cpp Authors
#include "subsetMNIST.h"

namespace {
constexpr uint32_t kTrainSize = 2000;
constexpr uint32_t kTestSize = 500;
constexpr uint32_t kImageMagicNumber = 2051;
constexpr uint32_t kTargetMagicNumber = 2049;
constexpr uint32_t kImageRows = 28;
constexpr uint32_t kImageColumns = 28;
constexpr const char* kTrainImagesFilename = "train-images-idx3-ubyte";
constexpr const char* kTrainTargetsFilename = "train-labels-idx1-ubyte";
constexpr const char* kTestImagesFilename = "t10k-images-idx3-ubyte";
constexpr const char* kTestTargetsFilename = "t10k-labels-idx1-ubyte";

bool check_is_little_endian() {
  const uint32_t word = 1;
  return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
}

constexpr uint32_t flip_endianness(uint32_t value) {
  return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u) |
      ((value & 0xff0000u) >> 8u) | ((value & 0xff000000u) >> 24u);
}

uint32_t read_int32(std::ifstream& stream) {
  static const bool is_little_endian = check_is_little_endian();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t value;
  AT_ASSERT(stream.read(reinterpret_cast<char*>(&value), sizeof value));
  return is_little_endian ? flip_endianness(value) : value;
}

uint32_t expect_int32(std::ifstream& stream, uint32_t expected) {
  const auto value = read_int32(stream);
  // clang-format off
  TORCH_CHECK(value == expected,
      "Expected to read number ", expected, " but found ", value, " instead");
  // clang-format on
  return value;
}

std::string join_paths(std::string head, const std::string& tail) {
  if (head.back() != '/') {
    head.push_back('/');
  }
  head += tail;
  return head;
}
/*
torch::Tensor read_images(const std::string& root, bool train) {
  const auto path = join_paths(root, train ? kTrainImagesFilename : kTestImagesFilename);
  std::ifstream images(path, std::ios::binary);
  TORCH_CHECK(images, "Error opening images file at ", path);

  // Expect the magic number for image files
  expect_int32(images, kImageMagicNumber);
  // Read the total number of images in the dataset (not using it for validation anymore)
  uint32_t total_count = read_int32(images); // Just read the value without checking
  expect_int32(images, kImageRows);
  expect_int32(images, kImageColumns);

  // Calculate the number of images to read (the minimum between total_count and kTrainSize or kTestSize)
  const auto count = train ? std::min(total_count, kTrainSize) : std::min(total_count, kTestSize);

  auto tensor = torch::empty({count, 1, kImageRows, kImageColumns}, torch::kByte);
  images.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());
  return tensor.to(torch::kFloat32).div_(255);
}

torch::Tensor read_targets(const std::string& root, bool train) {
  const auto path = join_paths(root, train ? kTrainTargetsFilename : kTestTargetsFilename);
  std::ifstream targets(path, std::ios::binary);
  TORCH_CHECK(targets, "Error opening targets file at ", path);

  // Expect the magic number for target files
  expect_int32(targets, kTargetMagicNumber);
  // Read the total number of targets in the dataset (not using it for validation anymore)
auto total_count = read_int32(targets);
  // Calculate the number of targets to read (the minimum between total_count and kTrainSize or kTestSize)
  const auto count = train ? std::min(total_count, kTrainSize) : std::min(total_count, kTestSize);

  auto tensor = torch::empty(count, torch::kByte);
  targets.read(reinterpret_cast<char*>(tensor.data_ptr()), count);
  return tensor.to(torch::kInt64);
}


torch::Tensor read_images(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainImagesFilename : kTestImagesFilename);
  std::ifstream images(path, std::ios::binary);
  TORCH_CHECK(images, "Error opening images file at ", path);

  const auto count = train ? kTrainSize : kTestSize;

  // From http://yann.lecun.com/exdb/mnist/
  expect_int32(images, kImageMagicNumber);
  expect_int32(images, count);
  expect_int32(images, kImageRows);
  expect_int32(images, kImageColumns);

  auto tensor =
      torch::empty({count, 1, kImageRows, kImageColumns}, torch::kByte);
  images.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());
  return tensor.to(torch::kFloat32).div_(255);
}

torch::Tensor read_targets(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainTargetsFilename : kTestTargetsFilename);
  std::ifstream targets(path, std::ios::binary);
  TORCH_CHECK(targets, "Error opening targets file at ", path);

  const auto count = train ? kTrainSize : kTestSize;

  expect_int32(targets, kTargetMagicNumber);
  expect_int32(targets, count);

  auto tensor = torch::empty(count, torch::kByte);
  targets.read(reinterpret_cast<char*>(tensor.data_ptr()), count);
  return tensor.to(torch::kInt64);
}

*/
torch::Tensor read_images(const std::string& root, bool train) {
  const auto path = join_paths(root, train ? kTrainImagesFilename : kTestImagesFilename);
  std::ifstream images(path, std::ios::binary);
  TORCH_CHECK(images, "Error opening images file at ", path);

  expect_int32(images, kImageMagicNumber); // Validate magic number
  uint32_t total_count = read_int32(images); // Read total number of images
  expect_int32(images, kImageRows); // Validate rows
  expect_int32(images, kImageColumns); // Validate columns

  // Determine the number of images to read based on the subset size
  const auto count = train ? kTrainSize : kTestSize;
  
  // Prepare tensor for the subset
  //auto tensor = torch::empty({count, 1, kImageRows, kImageColumns}, torch::kByte);
  std::cout<<"This is the size of the sampels"<<count;
 // Define the total number of pixels in one image
constexpr size_t num_pixels_per_image = kImageRows * kImageColumns;

// Calculate the total number of images to read based on the subset size

// Create a vector to store all image pixels before converting to a tensor
std::vector<uint8_t> image_pixels(count * num_pixels_per_image);

// Loop to read each image one by one
for(uint32_t i = 0; i < count; ++i) {
    // Calculate the start position for the current image in the vector
    auto start_pos = i * num_pixels_per_image;
    if (!images.read(reinterpret_cast<char*>(image_pixels.data() + start_pos), num_pixels_per_image)) {
        throw std::runtime_error("Failed to read image data from the file.");
    }
}

// Convert the entire vector of image pixels into a tensor at once
auto tensor = torch::from_blob(image_pixels.data(), {count, 1, kImageRows, kImageColumns}, torch::kUInt8)
                .to(torch::kFloat32).div_(255.0);

  return tensor.to(torch::kFloat32).div_(255);
}
torch::Tensor read_targets(const std::string& root, bool train) {
  const auto path = join_paths(root, train ? kTrainTargetsFilename : kTestTargetsFilename);
  std::ifstream targets(path, std::ios::binary);
  TORCH_CHECK(targets, "Error opening targets file at ", path);

  expect_int32(targets, kTargetMagicNumber); // Validate magic number
  uint32_t total_count = read_int32(targets); // Read total number of targets
  
  // Determine the number of targets to read based on the subset size
  const auto count = train ? kTrainSize : kTestSize;

  // Prepare tensor for the subset
  //auto tensor = torch::empty(count, torch::kByte);
  
  std::vector<int64_t> target_values(count);
for (uint32_t i = 0; i < count; ++i) {
    char label;
    if (!targets.read(&label, 1)) {
        throw std::runtime_error("Failed to read target data from the file.");
    }
    // Convert the read character to int64_t for consistency with PyTorch tensors
    target_values[i] = static_cast<int64_t>(static_cast<uint8_t>(label));
}

// Convert the entire vector of target values into a tensor at once
auto tensor = torch::tensor(target_values, torch::dtype(torch::kInt64));


  return tensor.to(torch::kInt64);
}


}  // namespace
/*
subsetMNIST::subsetMNIST(const std::string& root, Mode mode)
    : images_(read_images(root, mode == Mode::kTrain)),
      targets_(read_targets(root, mode == Mode::kTrain)) {}
*/

subsetMNIST::subsetMNIST(const std::string& root, Mode mode)
    : MNIST(root, mode == kTrain ? MNIST::Mode::kTrain : MNIST::Mode::kTest) { // Fixing mode type
    // Custom loading for subset
    images_ = read_images(root, mode == kTrain);
    targets_ = read_targets(root, mode == kTrain);
}
torch::data::Example<> subsetMNIST::get(size_t index) {
  return {images_[index], targets_[index]};
}

torch::optional<size_t> subsetMNIST::size() const {
  //return images_.size(0);
  return is_train() ? kTrainSize : kTestSize;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
bool subsetMNIST::is_train() const noexcept {
  return images_.size(0) == kTrainSize;
}

const torch::Tensor& subsetMNIST::images() const {
  return images_;
}

const torch::Tensor& subsetMNIST::targets() const {
  return targets_;
}
