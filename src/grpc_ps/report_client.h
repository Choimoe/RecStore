#pragma once
#include <string>

extern "C" {
    bool report(
        const char* table_name,
        const char* unique_id,
        const char* metric_name,
        double metric_value
    );
}