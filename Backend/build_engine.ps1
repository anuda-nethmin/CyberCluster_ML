if (!(Test-Path "build")) { New-Item -ItemType Directory -Path "build" -Force | Out-Null }

# Message to host
Write-Host "Compiling C Backend..." -ForegroundColor Cyan

# Compile all .c files to .o objects

gcc -Wall -Wextra -O3 -c main.c -o build/main.o
gcc -Wall -Wextra -O3 -c csv_parser.c -o build/csv_parser.o
gcc -Wall -Wextra -O3 -c linear_regression.c -o build/linear_regression.o
gcc -Wall -Wextra -O3 -c logistic_regression.c -o build/logistic_regression.o
gcc -Wall -Wextra -O3 -c knn.c -o build/knn.o
gcc -Wall -Wextra -O3 -c decision_tree.c -o build/decision_tree.o


if ($?) {
    # Link them all together into the final executable with the math library (-lm)
    gcc -Wall -Wextra -O3 -o build/ml_engine.exe build/main.o build/csv_parser.o build/linear_regression.o build/logistic_regression.o build/knn.o build/decision_tree.o -lm
    
    if ($?) {
        Write-Host "Build Complete! Output saved to build/ml_engine.exe" -ForegroundColor Green
    } else {
        Write-Host "Linking failed." -ForegroundColor Red
    }
} else {
    Write-Host "Compilation failed." -ForegroundColor Red
}