#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" cudaError_t compute_snp_contributions_cuda(
    const unsigned char* h_genotypes_batch,
    const float* h_scores_batch,
    float* h_contributions_out,
    int k,
    int N);

struct Options {
    int snps = 4096;
    int individuals = 2048;
    int batch_size = 64;
    int warmup = 1;
    int repeats = 5;
    unsigned int seed = 42;
};

struct Dataset {
    std::vector<unsigned char> genotypes;
    std::vector<float> scores;
};

void PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [--snps=N] [--individuals=N] [--batch-size=N] [--warmup=N]"
                 " [--repeats=N] [--seed=N]\n";
}

Options ParseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        auto pos = arg.find('=');
        if (pos == std::string::npos) {
            std::cerr << "Unrecognized argument: " << arg << "\n";
            PrintUsage(argv[0]);
            std::exit(1);
        }
        std::string key = arg.substr(0, pos);
        int value = std::stoi(arg.substr(pos + 1));
        if (key == "--snps") {
            opts.snps = value;
        } else if (key == "--individuals") {
            opts.individuals = value;
        } else if (key == "--batch-size") {
            opts.batch_size = value;
        } else if (key == "--warmup") {
            opts.warmup = value;
        } else if (key == "--repeats") {
            opts.repeats = value;
        } else if (key == "--seed") {
            opts.seed = static_cast<unsigned int>(value);
        } else {
            std::cerr << "Unknown flag: " << key << "\n";
            PrintUsage(argv[0]);
            std::exit(1);
        }
    }
    return opts;
}

Dataset GenerateDataset(const Options& opts) {
    if (opts.snps <= 0 || opts.individuals <= 0) {
        throw std::invalid_argument("snps and individuals must be positive");
    }

    Dataset data;
    const std::size_t total_genotypes =
        static_cast<std::size_t>(opts.snps) * static_cast<std::size_t>(opts.individuals);
    data.genotypes.resize(total_genotypes);
    data.scores.resize(static_cast<std::size_t>(opts.snps));

    std::mt19937 rng(opts.seed);
    std::discrete_distribution<int> geno_dist({70, 20, 5, 5});
    std::uniform_real_distribution<float> score_dist(0.05f, 1.5f);

    for (std::size_t i = 0; i < total_genotypes; ++i) {
        int allele = geno_dist(rng);
        static const unsigned char lookup[4] = {0u, 1u, 2u, 9u};
        data.genotypes[i] = lookup[allele];
    }
    for (float& s : data.scores) {
        s = score_dist(rng);
    }

    return data;
}

double RunPass(
    const Options& opts,
    const Dataset& data,
    std::vector<float>& workspace) {
    const int N = opts.individuals;
    const std::size_t stride = static_cast<std::size_t>(N);
    std::size_t offset = 0;

    const auto start = std::chrono::high_resolution_clock::now();

    while (offset < data.scores.size()) {
        const int remaining = static_cast<int>(data.scores.size() - offset);
        const int k = std::min(opts.batch_size, remaining);
        const unsigned char* geno_batch = data.genotypes.data() + offset * stride;
        const float* score_batch = data.scores.data() + offset;
        float* out_ptr = workspace.data();

        cudaError_t err = compute_snp_contributions_cuda(
            geno_batch,
            score_batch,
            out_ptr,
            k,
            N);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Kernel launch failed: ") + cudaGetErrorString(err));
        }

        offset += static_cast<std::size_t>(k);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

void PrintSummary(
    const Options& opts,
    const std::vector<double>& samples,
    double bytes_written) {
    if (samples.empty()) {
        std::cerr << "No measurement samples collected.\n";
        return;
    }

    const double best = *std::min_element(samples.begin(), samples.end());
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    double variance = 0.0;
    for (double v : sorted) {
        variance += (v - mean) * (v - mean);
    }
    variance /= sorted.size();
    const double stdev = std::sqrt(variance);

    const double snps_per_second = static_cast<double>(opts.snps) / best;
    const double gigabytes = bytes_written / 1e9;
    const double bandwidth = gigabytes / best;

    std::cout << "\nBenchmark summary\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Dataset   : " << opts.snps << " SNPs × " << opts.individuals
              << " individuals\n";
    std::cout << "Batch size: " << opts.batch_size << " SNPs per launch\n";
    std::cout << "Warmup    : " << opts.warmup << " pass(es)\n";
    std::cout << "Samples   : " << samples.size() << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Best      : " << best << " s\n";
    std::cout << "Median    : " << median << " s\n";
    std::cout << "Mean      : " << mean << " s\n";
    std::cout << "Stdev     : " << stdev << " s\n";
    std::cout << "SNP/s     : " << std::setprecision(0) << snps_per_second << "\n";
    std::cout << std::setprecision(3);
    std::cout << "Output    : " << gigabytes << " GB per pass\n";
    std::cout << "BW        : " << bandwidth << " GB/s effective\n";
    std::cout << "----------------------------------------\n";
}

int main(int argc, char** argv) {
    try {
        Options opts = ParseArgs(argc, argv);
        opts.batch_size = std::max(1, std::min(opts.batch_size, opts.snps));
        opts.warmup = std::max(0, opts.warmup);
        opts.repeats = std::max(1, opts.repeats);

        std::cout << "Generating synthetic dataset..." << std::endl;
        Dataset data = GenerateDataset(opts);

        const std::size_t workspace_elems =
            static_cast<std::size_t>(opts.batch_size) *
            static_cast<std::size_t>(opts.individuals) *
            static_cast<std::size_t>(opts.individuals);
        if (workspace_elems == 0) {
            throw std::runtime_error("Workspace allocation underflow");
        }
        std::vector<float> workspace(workspace_elems);
        const double bytes_written =
            static_cast<double>(opts.snps) *
            static_cast<double>(opts.individuals) *
            static_cast<double>(opts.individuals) *
            sizeof(float);

        for (int i = 0; i < opts.warmup; ++i) {
            std::cout << "Warmup pass " << (i + 1) << "/" << opts.warmup << "...\n";
            RunPass(opts, data, workspace);
        }

        std::vector<double> samples;
        samples.reserve(opts.repeats);
        for (int i = 0; i < opts.repeats; ++i) {
            const double elapsed = RunPass(opts, data, workspace);
            samples.push_back(elapsed);
            std::cout << "Measurement " << (i + 1) << "/" << opts.repeats << ": "
                      << std::fixed << std::setprecision(3) << elapsed << " s\n";
        }

        PrintSummary(opts, samples, bytes_written);
    } catch (const std::exception& ex) {
        std::cerr << "Benchmark failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}

