#pragma once

#include <cstdint>
#include <string>

namespace tsy {

struct SourceLocation {
    std::string file;
    int line = 0;
    int column = 0;

    bool valid() const { return line > 0; }

    std::string format() const {
        if (!valid()) return "<unknown>";
        std::string out = file.empty() ? std::string("<input>") : file;
        out += ":";
        out += std::to_string(line);
        out += ":";
        out += std::to_string(column);
        return out;
    }
};

}  // namespace tsy
