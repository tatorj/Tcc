/* Tests for final version of Luiz Otávio's TCC */

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

