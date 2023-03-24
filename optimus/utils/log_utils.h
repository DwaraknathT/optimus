#pragma once
#include <assert.h>
#include <iostream>
#include <string>

namespace optimus {

enum LogLevel { INFO, ERROR, WARNING };

inline void OPT_LOG(const std::string message, const LogLevel log_level) {
    if (log_level == LogLevel::INFO) {
        std::cout << "\nINFO: " << message << std::endl;
    } else if (log_level == LogLevel::ERROR) {
        std::cout << "\nERROR: " << message << std::endl;
    } else if (log_level == LogLevel::WARNING) {
        std::cout << "\nWARNING: " << message << std::endl;
    } else {
        std::cout << std::endl << message << std::endl;
    }
}

inline void OPT_CHECK(const bool condition, const std::string message) {
    if (condition != true) {
        OPT_LOG(message, LogLevel::ERROR);
        assert(condition == true);
    }
}

}  // namespace optimus