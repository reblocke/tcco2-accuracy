// Compress quadrature-weighted normal components into local Hermite expansions.
// Compare cached densities with direct evaluation before using a new cache.
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

extern "C" int cached_density(
    const double* draws, int count, const double* nodes, const double* weights,
    int order, const double* offsets, int points, double* output
) {
    constexpr int degree = 20;
    constexpr int scale_bins = 16;
    using Coefficients = std::array<double, degree + 1>;
    std::map<std::pair<int, int>, Coefficients> coefficients;

    // Map nodes remain stable. This small direct cache avoids repeated tree searches.
    constexpr int cache_size = 16384;
    struct Entry {
        int scale;
        int location;
        double* coefficients;
    };
    std::vector<Entry> cache(cache_size);
    Coefficients inverse_root{}, recurrence{};
    for (int k = 1; k <= degree; k++) {
        inverse_root[k] = 1 / std::sqrt(double(k));
        recurrence[k] = std::sqrt(double(k - 1) / k);
    }

    for (int i = 0; i < count; i++) {
        const double* theta = draws + 5 * i;
        double loading = theta[3] == 0 ? 0 : theta[2] * theta[4];
        double residual = theta[2] * theta[2] - loading * loading;
        for (int j = 0; j < order; j++) {
            double mean = theta[0] + loading * nodes[j];
            double variance = std::exp(theta[1] + theta[3] * nodes[j]) + residual;
            double sd = std::sqrt(variance);
            if (!(sd > 0 && std::isfinite(sd) && std::isfinite(mean))) {
                return 1;
            }
            int exponent;
            double fraction = std::frexp(sd, &exponent);
            int bin = int((fraction - 0.5) * (2 * scale_bins));
            double scale = std::ldexp(0.5 + (bin + 0.5) / (2 * scale_bins), exponent);
            double location = std::round(mean / (scale * 0.1));
            if (std::abs(location) > std::numeric_limits<int>::max()) {
                return 1;
            }
            int scale_key = exponent * scale_bins + bin;
            int location_key = int(location);
            auto& entry = cache[
                (uint32_t(scale_key) * 2654435761u + uint32_t(location_key)) % cache_size
            ];
            if (!entry.coefficients || entry.scale != scale_key || entry.location != location_key) {
                entry = {scale_key, location_key, coefficients[{scale_key, location_key}].data()};
            }
            double* cell = entry.coefficients;
            double displacement = mean / scale - 0.1 * location_key;
            double variance_delta = variance / (scale * scale) - 1;
            double previous = weights[j];
            double current = previous * displacement;
            cell[0] += previous;
            cell[1] += current;
            for (int k = 2; k <= degree; k++) {
                double next = displacement * inverse_root[k] * current
                    + variance_delta * recurrence[k] * previous;
                cell[k] += next;
                previous = current;
                current = next;
            }
        }
    }

    for (const auto& cell : coefficients) {
        int scale_key = cell.first.first;
        int exponent = int(std::floor(double(scale_key) / scale_bins));
        double scale = std::ldexp(
            0.5 + (scale_key - exponent * scale_bins + 0.5) / (2 * scale_bins), exponent
        );
        double location = cell.first.second * 0.1 * scale;
        for (int j = 0; j < points; j++) {
            double z = (offsets[j] - location) / scale;
            double previous = std::exp(-0.5 * z * z) / (2.5066282746310005 * scale);
            double current = previous * z;
            if (previous == 0) {
                continue;
            }
            double sum = cell.second[0] * previous + cell.second[1] * current;
            for (int k = 2; k <= degree; k++) {
                double next = inverse_root[k] * z * current - recurrence[k] * previous;
                sum += cell.second[k] * next;
                previous = current;
                current = next;
            }
            output[j] += sum / count;
        }
    }
    return 0;
}
