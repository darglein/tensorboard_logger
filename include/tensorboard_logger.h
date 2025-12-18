#ifndef TENSORBOARD_LOGGER_H
#define TENSORBOARD_LOGGER_H

#include <exception>
#include <fstream>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

#if defined _WIN32 || defined __CYGWIN__
#    define TB_HELPER_DLL_IMPORT __declspec(dllimport)
#    define TB_HELPER_DLL_EXPORT __declspec(dllexport)
#    define TB_HELPER_DLL_LOCAL
#else
#    if __GNUC__ >= 4  // Note: Clang also defines GNUC
#        define TB_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#        define TB_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#        define TB_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#    else
#        error Unknown import/export defines.
#        define TB_HELPER_DLL_IMPORT
#        define TB_HELPER_DLL_EXPORT
#        define TB_HELPER_DLL_LOCAL
#    endif
#endif


#if defined(tensorboard_logger_EXPORTS)
#    define TB_API TB_HELPER_DLL_EXPORT
#else
#    define TB_API TB_HELPER_DLL_IMPORT
#endif

namespace tensorflow {
class Event;
class Summary;
}
// extract parent dir or basename from path by finding the last slash
std::string get_parent_dir(const std::string &path);
std::string get_basename(const std::string &path);



struct TensorBoardLoggerOptions
{
    // Log is flushed whenever this many entries have been written since the last
    // forced flush.
    size_t max_queue_size_ = 100000;
    TensorBoardLoggerOptions &max_queue_size(size_t max_queue_size) {
        max_queue_size_ = max_queue_size;
        return *this;
    }

    // Log is flushed with this period.
    size_t flush_period_s_ = 60;
    TensorBoardLoggerOptions &flush_period_s(size_t flush_period_s) {
        flush_period_s_ = flush_period_s;
        return *this;
    }

    bool resume_ = false;
    TensorBoardLoggerOptions &resume(bool resume) {
        resume_ = resume;
        return *this;
    }
};

class TB_API TensorBoardLogger {
   public:
    
    explicit TensorBoardLogger(const std::string &log_file,
                               const TensorBoardLoggerOptions &options={}) {
        this->options = options;
        auto basename = get_basename(log_file);
        if (basename.find("tfevents") == std::string::npos) {
            throw std::runtime_error(
                "A valid event file must contain substring \"tfevents\" in its "
                "basename, got " + basename);
        }
        bucket_limits_ = nullptr;
        ofs_ = new std::ofstream(
            log_file, std::ios::out |
                          (options.resume_ ? std::ios::app : std::ios::trunc) |
                          std::ios::binary);
        if (!ofs_->is_open()) {
            throw std::runtime_error("failed to open log_file " + log_file);
        }
        log_dir_ = get_parent_dir(log_file);

        flushing_thread = std::thread(&TensorBoardLogger::flusher, this);
    }
    ~TensorBoardLogger() {
        ofs_->close();
        delete ofs_;
        ofs_ = nullptr;
        if (bucket_limits_ != nullptr) {
            delete bucket_limits_;
            bucket_limits_ = nullptr;
        }

        stop = true;
        if (flushing_thread.joinable()) {
            flushing_thread.join();
        }
    }
    int add_scalar(const std::string &tag, int step, double value);
    int add_scalar(const std::string &tag, int step, float value);

    // https://github.com/dmlc/tensorboard/blob/master/python/tensorboard/summary.py#L127
    int add_histogram(const std::string &tag, int step, const double *value,
                      size_t num);
    ;

    template <typename T>
    int add_histogram(const std::string &tag, int step,
                      const std::vector<T> &values) {
        return add_histogram(tag, step, values.data(), values.size());
    };

    // metadata (such as display_name, description) of the same tag will be
    // stripped to keep only the first one.
    int add_image(const std::string &tag, int step,
                  const std::string &encoded_image, int height, int width,
                  int channel, const std::string &display_name = "",
                  const std::string &description = "");
    int add_images(const std::string &tag, int step,
                   const std::vector<std::string> &encoded_images, int height,
                   int width, const std::string &display_name = "",
                   const std::string &description = "");
    int add_audio(const std::string &tag, int step,
                  const std::string &encoded_audio, float sample_rate,
                  int num_channels, int length_frame,
                  const std::string &content_type,
                  const std::string &display_name = "",
                  const std::string &description = "");
    int add_text(const std::string &tag, int step, const char *text);

    // `tensordata` and `metadata` should be in tsv format, and should be
    // manually created before calling `add_embedding`
    //
    // `tensor_name` is mandated to differentiate tensors
    //
    // TODO add sprite image support
    int add_embedding(
        const std::string &tensor_name, const std::string &tensordata_path,
        const std::string &metadata_path = "",
        const std::vector<uint32_t> &tensor_shape = std::vector<uint32_t>(),
        int step = 1 /* no effect */);
    // write tensor to binary file
    int add_embedding(
        const std::string &tensor_name,
        const std::vector<std::vector<float>> &tensor,
        const std::string &tensordata_filename,
        const std::vector<std::string> &metadata = std::vector<std::string>(),
        const std::string &metadata_filename = "",
        int step = 1 /* no effect */);
    int add_embedding(
        const std::string &tensor_name, const float *tensor,
        const std::vector<uint32_t> &tensor_shape,
        const std::string &tensordata_filename,
        const std::vector<std::string> &metadata = std::vector<std::string>(),
        const std::string &metadata_filename = "",
        int step = 1 /* no effect */);

   private:
    int generate_default_buckets();
    int add_event(int64_t step, tensorflow::Summary *summary);
    int write(tensorflow::Event &event);
    void flusher();

    std::string log_dir_;
    std::ofstream *ofs_;
    std::vector<double> *bucket_limits_;
    TensorBoardLoggerOptions options;
    
    std::atomic<bool> stop{false};
    size_t queue_size{0};
    std::thread flushing_thread;
    std::mutex file_object_mtx{};
};  // class TensorBoardLogger

#endif  // TENSORBOARD_LOGGER_H
