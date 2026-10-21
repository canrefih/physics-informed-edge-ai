#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());

    // Nearest-rank percentile, with rank starting at 1.
    const auto rank = static_cast<std::size_t>(
        std::ceil(p * values.size()));

    const auto index = std::max<std::size_t>(
        1, rank) - 1;

    return values[std::min(index, values.size() - 1)];
}

int main() {
    try {
        const std::filesystem::path model_path =
            "artifacts/model.onnx";

        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error(
                "Model not found: artifacts/model.onnx");
        }

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "RollomaticEdgeAI");

        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(env, model_path.c_str(), options);

        std::array<int64_t, 3> shape{1, 32, 2};
        std::vector<float> input(32 * 2);

        // Synthetic demonstration sensor window.
        for (int t = 0; t < 32; ++t) {
            const float temperature_delta =
                1.0f + 2.0f * t / 31.0f;
            const float vibration =
                0.15f + 0.10f * (t % 8) / 7.0f;

            input[t * 2] = temperature_delta / 10.0f;
            input[t * 2 + 1] = vibration;
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input.data(),
            input.size(),
            shape.data(),
            shape.size());

        const char* input_names[] = {"sensor_window"};
        const char* output_names[] = {"predicted_drift_mm"};

        auto run_inference = [&]() {
            return session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1);
        };

        constexpr int warmup_iterations = 100;
        constexpr int iterations = 5000;

        for (int i = 0; i < warmup_iterations; ++i) {
            auto output = run_inference();
        }

        std::vector<double> latency_us;
        latency_us.reserve(iterations);

        float predicted_drift_mm = 0.0f;

        for (int i = 0; i < iterations; ++i) {
            const auto start = std::chrono::steady_clock::now();
            auto output = run_inference();
            const auto end = std::chrono::steady_clock::now();

            latency_us.push_back(
                std::chrono::duration<double, std::micro>(
                    end - start).count());

            predicted_drift_mm =
                output[0].GetTensorData<float>()[0];
        }

        const double average_us =
            std::accumulate(
                latency_us.begin(), latency_us.end(), 0.0)
            / latency_us.size();

        const double median_us = percentile(latency_us, 0.50);
        const double p95_us = percentile(latency_us, 0.95);
        const double p99_us = percentile(latency_us, 0.99);

        const auto [min_it, max_it] = std::minmax_element(
            latency_us.begin(), latency_us.end());

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n=== Rollomatic Edge AI | C++ Benchmark ===\n";
        std::cout << "Model: " << model_path.string() << '\n';
        std::cout << "Input shape: [1, 32, 2]\n";
        std::cout << "Predicted drift: "
                  << std::setprecision(6)
                  << predicted_drift_mm << " mm\n";
        std::cout << "Predicted drift: "
                  << predicted_drift_mm * 1000.0f << " um\n";

        std::cout << std::setprecision(3);
        std::cout << "Warm-up iterations: " << warmup_iterations << '\n';
        std::cout << "Measured iterations: " << iterations << '\n';
        std::cout << "Average latency: " << average_us << " us\n";
        std::cout << "Median latency:  " << median_us << " us\n";
        std::cout << "P95 latency:     " << p95_us << " us\n";
        std::cout << "P99 latency:     " << p99_us << " us\n";
        std::cout << "Min latency:     " << *min_it << " us\n";
        std::cout << "Max latency:     " << *max_it << " us\n";

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';
        return 1;
    }
}
