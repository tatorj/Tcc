/* Copyright 2022 Luiz Otavio Soares de Oliveira by FEN/UERJ
 * This file is part of the final version of the TCC by Luiz Otavio, as 
 * a requirement for obtaining a degree at this public university under 
 * guidance of Irving Badolato professor.
 * The resulting software is free: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License (GNU GPL) as 
 * published by the Free Software Foundation, either version 3 of the 
 * License, or (at your option) any later version.
 * Our code is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
 * Read the GNU GPL for more details. To obtain a copy of this license 
 * see <http://www.gnu.org/licenses/>.
 */

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
