# This is a "driver" makefile that, for convenience, replaces in-line variable setting with simple phony targets.
.PHONY: usage serial serial_dbg serial_prof pll pll_dbg pll_prof gpu gpu_dbg gpu_prof clean doc test

usage:
	@echo
	@echo "Available make targets:"
	@echo "make serial"
	@echo "make serial_dbg"
	@echo "make gpu"
	@echo
	@echo "make clean"
	@echo
	@echo "May build multiple rules concurrently, e.g., make -j4 serial gpu"

serial:
	make -f makefiles/Makefile

serial_prof:
	make BUILD=profile -f makefiles/Makefile

serial_dbg:
	make BUILD=debug -f makefiles/Makefile

pll:
	make PLL=pll -f makefiles/Makefile

gpu:
	make PLL=gpu -f makefiles/Makefile

gpu_dbg:
	make PLL=gpu BUILD=debug -f makefiles/Makefile

staticcheck:
	cd src; cppcheck --enable=all -f --inconclusive --std=c++11 --std=posix * > ../cppcheck.out 2>&1

test:
	make BUILD=debug -f makefiles/Makefile test/test.x test/testpot.x test/testGE.x
	cd test; ./test.x

inttest:
	make BUILD=debug -f makefiles/Makefile
	cd test/integration; ./run_testsuite.sh

GPUtest:
	make PLL=gpu BUILD=debug -f makefiles/Makefile test/test.x

coverage:
	make BUILD=coverage -f makefiles/Makefile clean
	rm -f *.gcda *.gcno
	make BUILD=coverage -f makefiles/Makefile test/test.x
	cd ./test; ./test.x
	gcov test/unittests.cpp -o .
	lcov -c --directory . --output-file ./test/coverage.info
	genhtml --legend --demangle-cpp --output-directory ./test/coverage_html ./test/coverage.info


clean:
	make -f makefiles/Makefile clean
	make BUILD=debug -f makefiles/Makefile clean
	make BUILD=coverage -f makefiles/Makefile clean
#	make BUILD=gpu -f makefiles/Makefile clean
