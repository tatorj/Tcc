/* Tests for final version of Luiz Otávio's TCC */

// Metric macros
#define GET_P_TIME(p) ({\
    auto t0 = std::chrono::high_resolution_clock::now();\
    p;\
    auto t1 = std::chrono::high_resolution_clock::now();\
    t1 - t0; })

#define GET_F_TIME(r, f) ({\
    auto t0 = std::chrono::high_resolution_clock::now();\
    r = f;\
    auto t1 = std::chrono::high_resolution_clock::now();\
    t1 - t0; })

#define GET_VECTOR_USAGE(vec) ({\
    vec.size() * sizeof(vec.front()); })

#define GET_2DVECTOR_USAGE(vec) ({\
    size_t sum = 0;\
    for (auto it = vec.begin(); it < vec.end(); it++)\
    sum += (*it).size() * sizeof( (*it).front() );\
    sum; })

#define GET_CVMAT_USAGE(mat) ({\
    (mat.cols*mat.rows) * sizeof(mat.type()); })

#define HUMAN_READABLE(m_usage) ({\
    std::string suffix[] = {"B", "KB", "MB", "GB", "TB"};\
    char length = sizeof(suffix) / sizeof(suffix[0]);\
    size_t bytes = m_usage;\
    double dblBytes = bytes;\
    unsigned char i = 0;\
    if (bytes > 1024)\
    for (; bytes / 1024 > 0 && i < length - 1; i++, bytes /= 1024)\
    dblBytes = bytes / 1024.0;\
    std::string result = std::to_string(dblBytes);\
    result.substr(0, result.size()-4) + " " + suffix[i]; })
