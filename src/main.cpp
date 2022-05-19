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

#include "control.hpp"

// Main routine
int main (int argc, char** argv) {
    lo::ProcessController controller;

    if ( controller.readArguments(argc, argv) && controller.runProcesses() )
        controller.saveResults();
    else
        controller.printUsage();

    return 0;
}

/* Notes on the libraries adoption:
 * - We uses STL containers in this project: vectors mainly, but this 
 * project requires at least one list, as this structure guarantees that 
 * the iterators (or even simple pointers) will not be invalidated when 
 * a new element is added or even removed. Maps were also used to ensure 
 * a quick search for measurements taken on images and images by index 
 * and to get images by index on process controller.
 * - We uses openCV algorithms: some experimental or non-free opencv 
 * classes used for academic tests can be removed from compilation by 
 * default if opencv from the standard linux repository is in use.
 */
