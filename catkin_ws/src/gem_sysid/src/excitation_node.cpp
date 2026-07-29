#include <ackermann_msgs/AckermannDrive.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Header.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ProfileSample {
  double time_s;
  double speed_mps;
  double steering_rad;
};

constexpr std::size_t kLowerCommandCount = 6;

const std::vector<std::string> kDefaultLowerCommandTopics = {
    "/gem/left_steering_ctrlr/command",
    "/gem/right_steering_ctrlr/command",
    "/gem/left_front_wheel_ctrlr/command",
    "/gem/right_front_wheel_ctrlr/command",
    "/gem/left_rear_wheel_ctrlr/command",
    "/gem/right_rear_wheel_ctrlr/command"};

std::string trim(const std::string& value) {
  const std::string whitespace = " \t\r\n";
  const std::size_t first = value.find_first_not_of(whitespace);
  if (first == std::string::npos) {
    return "";
  }
  const std::size_t last = value.find_last_not_of(whitespace);
  return value.substr(first, last - first + 1);
}

std::vector<std::string> splitCsvRow(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(trim(field));
  }
  return fields;
}

double parseNumber(const std::string& value, std::size_t line_number) {
  try {
    std::size_t consumed = 0;
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
      throw std::invalid_argument("not a finite number");
    }
    return parsed;
  } catch (const std::exception&) {
    throw std::runtime_error(
        "Invalid numeric value on CSV line " + std::to_string(line_number));
  }
}

std::vector<ProfileSample> loadProfile(
    const std::string& path,
    double sample_period_s,
    double max_speed_mps,
    double max_abs_steering_rad) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Cannot open profile CSV: " + path);
  }

  std::string line;
  if (!std::getline(input, line)) {
    throw std::runtime_error("Profile CSV is empty: " + path);
  }
  const std::vector<std::string> header = splitCsvRow(line);
  const std::vector<std::string> expected_header = {
      "time_s", "speed_mps", "steering_rad"};
  if (header != expected_header) {
    throw std::runtime_error(
        "Profile CSV header must be: time_s,speed_mps,steering_rad");
  }

  std::vector<ProfileSample> samples;
  std::size_t line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    if (trim(line).empty()) {
      continue;
    }
    const std::vector<std::string> fields = splitCsvRow(line);
    if (fields.size() != 3) {
      throw std::runtime_error(
          "Expected three CSV fields on line " + std::to_string(line_number));
    }
    const ProfileSample sample{
        parseNumber(fields[0], line_number),
        parseNumber(fields[1], line_number),
        parseNumber(fields[2], line_number)};

    if (sample.time_s < 0.0) {
      throw std::runtime_error("Profile timestamps must be nonnegative");
    }
    if (std::abs(sample.speed_mps) > max_speed_mps + 1e-9) {
      throw std::runtime_error(
          "Profile speed exceeds max_speed_mps on line " +
          std::to_string(line_number));
    }
    if (std::abs(sample.steering_rad) > max_abs_steering_rad + 1e-9) {
      throw std::runtime_error(
          "Profile steering exceeds max_abs_steering_rad on line " +
          std::to_string(line_number));
    }
    if (!samples.empty()) {
      const double interval = sample.time_s - samples.back().time_s;
      if (std::abs(interval - sample_period_s) > 1e-6) {
        throw std::runtime_error(
            "Profile timestamps must use the configured sample period on line " +
            std::to_string(line_number));
      }
    }
    samples.push_back(sample);
  }

  if (samples.empty()) {
    throw std::runtime_error("Profile CSV contains no samples");
  }
  if (std::abs(samples.front().time_s) > 1e-9) {
    throw std::runtime_error("The first profile timestamp must be 0.0");
  }
  const auto is_stopped = [](const ProfileSample& sample) {
    return std::abs(sample.speed_mps) < 1e-9 &&
           std::abs(sample.steering_rad) < 1e-9;
  };
  if (!is_stopped(samples.front()) || !is_stopped(samples.back())) {
    throw std::runtime_error(
        "The first and last profile samples must command a complete stop");
  }
  return samples;
}

ros::Time timeFromSeconds(double seconds) {
  ros::Time time;
  time.fromSec(seconds);
  return time;
}

void sleepUntil(const ros::Time& target_time) {
  while (ros::ok() && ros::Time::now() < target_time) {
    const double remaining = (target_time - ros::Time::now()).toSec();
    ros::Duration(std::min(remaining, 0.005)).sleep();
  }
}

struct LowerCommandBatch {
  ros::Time center_time;
  std::array<double, kLowerCommandCount> values;

  bool matchesSteering(double steering_rad) const {
    constexpr double kZeroTolerance = 1e-4;
    constexpr double kNonzeroTolerance = 1e-5;
    if (std::abs(steering_rad) < kZeroTolerance) {
      return std::abs(values[0]) < kZeroTolerance &&
             std::abs(values[1]) < kZeroTolerance;
    }
    return values[0] * steering_rad > 0.0 &&
           values[1] * steering_rad > 0.0 &&
           std::abs(values[0]) > kNonzeroTolerance &&
           std::abs(values[1]) > kNonzeroTolerance;
  }
};

class LowerCommandMonitor {
 public:
  LowerCommandMonitor(
      ros::NodeHandle& node,
      const std::vector<std::string>& topics,
      double max_batch_span_s)
      : max_batch_span_s_(max_batch_span_s), batch_active_(false) {
    if (topics.size() != kLowerCommandCount) {
      throw std::runtime_error(
          "lower_command_topics must contain exactly six topics");
    }
    seen_.fill(false);
    receipt_times_.fill(0.0);
    values_.fill(0.0);
    subscribers_.reserve(topics.size());
    for (std::size_t index = 0; index < topics.size(); ++index) {
      subscribers_.push_back(node.subscribe<std_msgs::Float64>(
          topics[index],
          10,
          [this, index](const std_msgs::Float64::ConstPtr& message) {
            commandCallback(message, index);
          }));
    }
  }

  bool waitForMatchingBatch(
      const ros::Time& after,
      double steering_rad,
      double wall_timeout_s,
      LowerCommandBatch* result) const {
    const ros::WallTime deadline =
        ros::WallTime::now() + ros::WallDuration(wall_timeout_s);
    while (ros::ok() && ros::WallTime::now() < deadline) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const LowerCommandBatch& batch : complete_batches_) {
          if (batch.center_time > after &&
              batch.matchesSteering(steering_rad)) {
            *result = batch;
            return true;
          }
        }
      }
      ros::WallDuration(0.001).sleep();
    }
    return false;
  }

 private:
  void commandCallback(
      const std_msgs::Float64::ConstPtr& message,
      std::size_t index) {
    const double receipt_time = ros::Time::now().toSec();
    std::lock_guard<std::mutex> lock(mutex_);

    const bool batch_expired =
        batch_active_ &&
        receipt_time - first_receipt_time_ > max_batch_span_s_;
    if (!batch_active_ || batch_expired || seen_[index]) {
      startBatch(receipt_time);
    }

    seen_[index] = true;
    receipt_times_[index] = receipt_time;
    values_[index] = message->data;

    if (std::all_of(seen_.begin(), seen_.end(), [](bool value) {
          return value;
        })) {
      double center_seconds = 0.0;
      for (double time : receipt_times_) {
        center_seconds += time;
      }
      center_seconds /= static_cast<double>(kLowerCommandCount);
      complete_batches_.push_back(
          LowerCommandBatch{timeFromSeconds(center_seconds), values_});
      if (complete_batches_.size() > 200) {
        complete_batches_.pop_front();
      }
      batch_active_ = false;
      seen_.fill(false);
    }
  }

  void startBatch(double receipt_time) {
    batch_active_ = true;
    first_receipt_time_ = receipt_time;
    seen_.fill(false);
  }

  double max_batch_span_s_;
  std::vector<ros::Subscriber> subscribers_;
  mutable std::mutex mutex_;
  mutable std::deque<LowerCommandBatch> complete_batches_;
  bool batch_active_;
  double first_receipt_time_;
  std::array<bool, kLowerCommandCount> seen_;
  std::array<double, kLowerCommandCount> receipt_times_;
  std::array<double, kLowerCommandCount> values_;
};

class ExcitationPublisher {
 public:
  ExcitationPublisher(ros::NodeHandle& node, ros::NodeHandle& private_node)
      : sequence_(0) {
    std::string command_topic;
    std::string stamped_command_topic;
    std::string profile_start_topic;
    std::string commissioning_delay_topic;
    private_node.param<std::string>(
        "command_topic", command_topic, "/gem/ackermann_cmd");
    private_node.param<std::string>(
        "stamped_command_topic",
        stamped_command_topic,
        "/gem_sysid/ackermann_cmd_stamped");
    private_node.param<std::string>(
        "profile_start_topic",
        profile_start_topic,
        "/gem_sysid/profile_start");
    private_node.param<std::string>(
        "commissioning_delay_topic",
        commissioning_delay_topic,
        "/gem_sysid/commissioning_delay_ms");
    command_publisher_ =
        node.advertise<ackermann_msgs::AckermannDrive>(command_topic, 1);
    stamped_publisher_ =
        node.advertise<ackermann_msgs::AckermannDriveStamped>(
            stamped_command_topic, 1);
    profile_start_publisher_ =
        node.advertise<std_msgs::Header>(profile_start_topic, 1, true);
    commissioning_delay_publisher_ =
        node.advertise<std_msgs::Float64>(
            commissioning_delay_topic, 1, true);
  }

  void waitForSubscribers(double timeout_s) const {
    const ros::WallTime deadline =
        ros::WallTime::now() + ros::WallDuration(timeout_s);
    while (ros::ok() && ros::WallTime::now() < deadline) {
      if (command_publisher_.getNumSubscribers() > 0 &&
          stamped_publisher_.getNumSubscribers() > 0) {
        return;
      }
      ros::WallDuration(0.05).sleep();
    }
    ROS_WARN(
        "Timed out waiting for all command subscribers; starting playback");
  }

  ros::Time publish(const ProfileSample& sample) {
    ackermann_msgs::AckermannDrive drive;
    drive.speed = static_cast<float>(sample.speed_mps);
    drive.steering_angle = static_cast<float>(sample.steering_rad);

    ackermann_msgs::AckermannDriveStamped stamped;
    stamped.header.seq = sequence_++;
    stamped.header.stamp = ros::Time::now();
    stamped.header.frame_id = "base_footprint";
    stamped.drive = drive;

    stamped_publisher_.publish(stamped);
    command_publisher_.publish(drive);
    return stamped.header.stamp;
  }

  void publishProfileStart(
      const ros::Time& start_time,
      double mean_delay_ms) {
    std_msgs::Header start;
    start.seq = 0;
    start.stamp = start_time;
    start.frame_id = "base_footprint";
    profile_start_publisher_.publish(start);

    std_msgs::Float64 delay;
    delay.data = mean_delay_ms;
    commissioning_delay_publisher_.publish(delay);
  }

  void publishStop() {
    const ProfileSample stop{0.0, 0.0, 0.0};
    for (int count = 0; count < 3 && ros::ok(); ++count) {
      publish(stop);
      ros::Duration(0.02).sleep();
    }
  }

 private:
  ros::Publisher command_publisher_;
  ros::Publisher stamped_publisher_;
  ros::Publisher profile_start_publisher_;
  ros::Publisher commissioning_delay_publisher_;
  std::uint32_t sequence_;
};

struct CommissioningResult {
  ros::Time profile_start_time;
  double mean_delay_ms;
};

CommissioningResult commissionCommandPhase(
    ExcitationPublisher& publisher,
    const LowerCommandMonitor& monitor,
    double sample_period_s,
    double target_delay_s,
    int trial_count,
    double marker_steering_rad,
    double response_timeout_s) {
  const ProfileSample stopped{0.0, 0.0, 0.0};
  const ros::Time baseline_publish_time = publisher.publish(stopped);
  LowerCommandBatch baseline_batch;
  if (!monitor.waitForMatchingBatch(
          baseline_publish_time,
          0.0,
          response_timeout_s,
          &baseline_batch)) {
    throw std::runtime_error(
        "Commissioning could not observe a complete zero-steering lower batch");
  }

  ros::Time next_publish_time =
      baseline_batch.center_time +
      ros::Duration(sample_period_s - target_delay_s);
  std::vector<double> delays_ms;
  delays_ms.reserve(static_cast<std::size_t>(trial_count));
  const double maximum_valid_delay_ms =
      1000.0 * target_delay_s + 5.0;
  const int maximum_attempt_count = 3 * trial_count;

  for (
      int attempt = 0;
      attempt < maximum_attempt_count &&
      static_cast<int>(delays_ms.size()) < trial_count;
      ++attempt) {
    const double steering =
        (attempt % 2 == 0) ? marker_steering_rad : -marker_steering_rad;
    sleepUntil(next_publish_time);
    if (!ros::ok()) {
      throw std::runtime_error("Commissioning interrupted");
    }

    const ProfileSample marker{0.0, 0.0, steering};
    const ros::Time publish_time = publisher.publish(marker);
    LowerCommandBatch takeover_batch;
    if (!monitor.waitForMatchingBatch(
            publish_time,
            steering,
            response_timeout_s,
            &takeover_batch)) {
      throw std::runtime_error(
          "Commissioning marker was not observed at the lower controllers");
    }

    const double delay_ms =
        1000.0 * (takeover_batch.center_time - publish_time).toSec();
    if (delay_ms >= 0.0 && delay_ms <= maximum_valid_delay_ms) {
      delays_ms.push_back(delay_ms);
      ROS_INFO(
          "Commissioning trial %zu/%d: lower-batch delay %.3f ms",
          delays_ms.size(),
          trial_count,
          delay_ms);
    } else {
      ROS_WARN(
          "Discarding commissioning attempt %d: missed lower-controller "
          "boundary (%.3f ms)",
          attempt + 1,
          delay_ms);
    }

    next_publish_time =
        takeover_batch.center_time +
        ros::Duration(sample_period_s - target_delay_s);
  }
  if (static_cast<int>(delays_ms.size()) < trial_count) {
    throw std::runtime_error(
        "Commissioning could not collect enough valid phase samples");
  }

  sleepUntil(next_publish_time);
  const ros::Time zero_publish_time = publisher.publish(stopped);
  LowerCommandBatch zero_batch;
  if (!monitor.waitForMatchingBatch(
          zero_publish_time,
          0.0,
          response_timeout_s,
          &zero_batch)) {
    throw std::runtime_error(
        "Commissioning could not confirm the final zero-steering command");
  }

  double sum_ms = 0.0;
  double minimum_ms = std::numeric_limits<double>::infinity();
  double maximum_ms = -std::numeric_limits<double>::infinity();
  for (double delay_ms : delays_ms) {
    sum_ms += delay_ms;
    minimum_ms = std::min(minimum_ms, delay_ms);
    maximum_ms = std::max(maximum_ms, delay_ms);
  }
  const double mean_ms = sum_ms / static_cast<double>(delays_ms.size());
  const double target_ms = 1000.0 * target_delay_s;
  ROS_INFO(
      "Commissioning complete: mean %.3f ms, min %.3f ms, max %.3f ms "
      "(target %.3f ms)",
      mean_ms,
      minimum_ms,
      maximum_ms,
      target_ms);
  if (
      std::abs(mean_ms - target_ms) > 2.0) {
    throw std::runtime_error(
        "Commissioning delays are not stable around the target");
  }

  return CommissioningResult{
      zero_batch.center_time +
          ros::Duration(sample_period_s - target_delay_s),
      mean_ms};
}

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "excitation_node");
  ros::NodeHandle node;
  ros::NodeHandle private_node("~");

  std::string profile_csv;
  double sample_period_s = 0.1;
  double max_speed_mps = 5.56;
  double max_abs_steering_rad = 0.30;
  double startup_delay_s = 1.0;
  double subscriber_timeout_s = 5.0;
  bool enable_phase_commissioning = true;
  double target_takeover_delay_s = 0.005;
  int commissioning_trials = 10;
  double commissioning_steering_rad = 0.02;
  double commissioning_batch_span_s = 0.010;
  double commissioning_response_timeout_s = 5.0;
  std::vector<std::string> lower_command_topics =
      kDefaultLowerCommandTopics;
  bool validate_only = false;
  private_node.param<std::string>("profile_csv", profile_csv, "");
  private_node.param("sample_period_s", sample_period_s, sample_period_s);
  private_node.param("max_speed_mps", max_speed_mps, max_speed_mps);
  private_node.param(
      "max_abs_steering_rad",
      max_abs_steering_rad,
      max_abs_steering_rad);
  private_node.param("startup_delay_s", startup_delay_s, startup_delay_s);
  private_node.param(
      "subscriber_timeout_s", subscriber_timeout_s, subscriber_timeout_s);
  private_node.param(
      "enable_phase_commissioning",
      enable_phase_commissioning,
      enable_phase_commissioning);
  private_node.param(
      "target_takeover_delay_s",
      target_takeover_delay_s,
      target_takeover_delay_s);
  private_node.param(
      "commissioning_trials",
      commissioning_trials,
      commissioning_trials);
  private_node.param(
      "commissioning_steering_rad",
      commissioning_steering_rad,
      commissioning_steering_rad);
  private_node.param(
      "commissioning_batch_span_s",
      commissioning_batch_span_s,
      commissioning_batch_span_s);
  private_node.param(
      "commissioning_response_timeout_s",
      commissioning_response_timeout_s,
      commissioning_response_timeout_s);
  private_node.getParam("lower_command_topics", lower_command_topics);
  private_node.param("validate_only", validate_only, validate_only);

  if (profile_csv.empty() || sample_period_s <= 0.0 ||
      max_speed_mps <= 0.0 || max_abs_steering_rad <= 0.0 ||
      startup_delay_s < 0.0 || subscriber_timeout_s < 0.0 ||
      target_takeover_delay_s <= 0.0 ||
      target_takeover_delay_s >= sample_period_s ||
      commissioning_trials <= 0 ||
      commissioning_steering_rad <= 0.0 ||
      commissioning_steering_rad > max_abs_steering_rad ||
      commissioning_batch_span_s <= 0.0 ||
      commissioning_response_timeout_s <= 0.0 ||
      lower_command_topics.size() != kLowerCommandCount) {
    ROS_FATAL("Invalid or missing excitation-node parameter");
    return 1;
  }

  std::vector<ProfileSample> samples;
  try {
    samples = loadProfile(
        profile_csv,
        sample_period_s,
        max_speed_mps,
        max_abs_steering_rad);
  } catch (const std::exception& error) {
    ROS_FATAL_STREAM(error.what());
    return 1;
  }

  ROS_INFO(
      "Loaded %zu commands covering %.1f seconds from %s",
      samples.size(),
      samples.back().time_s,
      profile_csv.c_str());
  if (validate_only) {
    ROS_INFO("Profile validation completed; no commands were published");
    return 0;
  }

  ExcitationPublisher publisher(node, private_node);
  LowerCommandMonitor monitor(
      node,
      lower_command_topics,
      commissioning_batch_span_s);
  ros::AsyncSpinner spinner(1);
  spinner.start();
  publisher.waitForSubscribers(subscriber_timeout_s);

  while (ros::ok() && ros::Time::now().isZero()) {
    ros::WallDuration(0.01).sleep();
  }
  if (!ros::ok()) {
    return 0;
  }

  ros::Duration(startup_delay_s).sleep();
  CommissioningResult commissioning{
      ros::Time::now(),
      std::numeric_limits<double>::quiet_NaN()};
  if (enable_phase_commissioning) {
    try {
      commissioning = commissionCommandPhase(
          publisher,
          monitor,
          sample_period_s,
          target_takeover_delay_s,
          commissioning_trials,
          commissioning_steering_rad,
          commissioning_response_timeout_s);
    } catch (const std::exception& error) {
      ROS_FATAL_STREAM(error.what());
      publisher.publishStop();
      return 1;
    }
  }

  const ros::Time start_time = enable_phase_commissioning
                                   ? commissioning.profile_start_time
                                   : ros::Time::now();
  publisher.publishProfileStart(start_time, commissioning.mean_delay_ms);
  ROS_INFO("Starting excitation playback at simulation time %.3f", start_time.toSec());

  for (const ProfileSample& sample : samples) {
    const ros::Time target_time = start_time + ros::Duration(sample.time_s);
    sleepUntil(target_time);
    if (!ros::ok()) {
      break;
    }
    publisher.publish(sample);
  }

  publisher.publishStop();
  ROS_INFO("Excitation playback complete; stop command published");
  return 0;
}
