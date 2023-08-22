##############################################################
# Set type of build
##############################################################
# Default to release build override with `make BUILD=debug`, `make BUILD=profile` or `make BUILD=coverage`
BUILD = release
# Default to serial build
PLL   = serial

##############################################################
# Import platform/architecture dependent settings, including:
# - Compiler and flags
# - Libraries
# - Includes
# In general, these will depend on both BUILD and PLL variables.
##############################################################
$(if $(wildcard makefiles/arch.mk),,$(error Error: Please create arch.mk in makefiles/ directory for machine-dependent configuration))
include makefiles/arch.mk

##############################################################
# Set up stub strings to avoid product name collisions
##############################################################
# Conditional assignment of the suffix for directories and products
ifeq ($(PLL),serial)
  suffix =
else ifeq ($(PLL),pll)
  suffix = PLL
else ifeq ($(PLL),gpu)
  suffix = GPU
else
  $(error Error: unknown option for PLL variable)
endif

##############################################################
# Specify product names
##############################################################
exe := Mott_Sum_Over_States$(suffix).$(if $(exesuffix),$(exesuffix),x)
bindir.debug   := bin/Debug
bindir.release := bin/Release
bindir.profile := bin/Profile
bindir.coverage := bin/Coverage
bindir := $(bindir.$(BUILD))

##############################################################
# Update suffix variable so that debug and release have distinct intermediates
##############################################################
ifeq ($(BUILD),debug)
  suffix := $(suffix)_dbg
else ifeq ($(BUILD),profile)
  suffix := $(suffix)_prof
else ifeq ($(BUILD),coverage)
  suffix := $(suffix)_cov
else ifneq ($(BUILD),release)
  $(error Error: unknown option for BUILD variable)
endif

##############################################################
# Import the list of source files
##############################################################
include makefiles/sources.mk

##############################################################
# Generate list of object files and target location
##############################################################
ifeq ($(PLL),serial)
  objdir  := obj/serial$(suffix)
else
  objdir  := obj/$(suffix)
endif
objects := $(addprefix $(objdir)/,$(sources:.cpp=.o))
ifeq ($(PLL),gpu)
  objects += $(addprefix $(objdir)/,$(gpuapisources:.cpp=.o))
  objects += $(addprefix $(objdir)/,$(cudasources:.cu=.o))
endif

################################################################################
########################### COMMON RULES FOR TARGETS ###########################
################################################################################

# Build targets
.PHONY: all # Phony targets do not correspond to explicit file products
all: $(bindir)/$(exe)

$(bindir)/$(exe): CMD = $(CXX) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -o $@ $(objects) $(srcdir)/main.cpp $(LDFLAGS)
$(bindir)/$(exe): $(objects) | $(bindir)
	@$(BUILD_CMD)

test/test.x: $(objects)
	$(CXX) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -I./src/ test/unittests.cpp $(objects) -o $@ $(LDFLAGS)

test/testpot.x: $(objects)
	$(CXX) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -I./src/ test/unittests_pot.cpp $(objects) -o $@ $(LDFLAGS)

test/testGE.x: $(objects)
	$(CXX) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -I./src/ test/test_gausselim.cpp $(objects) -o $@ $(LDFLAGS)

$(srcdir)/version.h: $(srcdir)/version.h_template
	@echo "Updating code version information"
	@sed s/hash/"$(shell git describe --always --long --abbrev=10 --dirty="-LocalModifications" --tags)"/g $< | sed s/-g/" "/g > $@.tmp
	@sed s/branch/"$(shell git branch | sed -n '/\* /s///p')"/g $@.tmp > $@
	@rm -f $@.tmp

# how to build any .o from the corresponding source
# the | order-only dependency is to create objdir on demand (but *without* allowing time-stamp changes in objdir to force recompilation of any dependent targets)
#
# First set the CMD variable, then process the target
#  note that we can't change make variables inside target rules (should be shell commands only)
#  but we can do it using dependency rule.
$(objdir)/%.o: CMD = $(CXX) -c $(ASMFLAGS) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -o $@ $<
$(objdir)/%.o: $(srcdir)/%.cpp $(objdir)/%.d | $(objdir) $(srcdir)/version.h
	@$(BUILD_CMD)
# Also for CUDA sources
$(objdir)/%.o: $(srcdir)/%.cu $(objdir)/%.d | $(objdir) $(srcdir)/version.h
	@$(BUILD_CMD)

.PRECIOUS: $(depends) # If make builds intermediates in chaining two rules it will delete them when the build is over. Prevent using .SECONDARY or .PRECIOUS
$(objdir)/%.d: $(srcdir)/%.cpp | $(objdir) $(srcdir)/version.h
	@echo "Generating dependency list for $(@:.d=.o)"
	@$(CPP) $< -std=c++11 $(DEFINE) $(INCLUDE) -MP -MM -MT $(@:.d=.o) -MF $@
# The previous line updates the dependency list whenever a file is rebuilt - makes detecting future changes to any new or existing dependencies robust.
# The -MP flag emits phony dependency targets for each included file. These avoid make errors if header files are removed or renamed; without
# this step the dependency files would have to be recreated, but Make has no way to determine that the dep list should be updated.
# Could be combined in this way, but fails for nvcc, so stick with separate dependency searching using cpp.

# Also do for CUDA sources
$(objdir)/%.d: $(srcdir)/%.cu | $(objdir)
	@echo "Generating dependency list for $(@:.d=.o)"
	@$(CPP) -x c++ $< $(DEFINE) $(INCLUDE) -MP -MM -MT $(@:.d=.o) -MF $@

# Import the dependency rules ... the - causes files that don't exist to not create an error
# We use -include because the .d files are only created during the first build and include will not find them.
depends := $(objects:.o=.d)
-include $(depends)

# To create folders on demand
$(objdir) $(bindir):
	mkdir -p $@

# Target to clean all objects
.PHONY: clean # Phony targets do not correspond to explicit file products
clean:
	rm -rf $(objects)

.PHONY: .FORCE
.FORCE: ;
$(srcdir)/version.h: .FORCE
$(bindir)/$(exe): .FORCE
test/test.x: .FORCE
test/testpot.x: .FORCE
test/testGE.x: .FORCE


################################################################################
########################### VISUAL STYLING OF OUTPUT ###########################
################################################################################
# Test for C++ compiler color output option
ifeq ("$(shell echo "int main(){}" | $(CXX) -fdiagnostics-color -x c - -o /dev/null 2>&1)", "")
  override CFLAGS += -fdiagnostics-color
else ifeq ("$(shell echo "int main(){}" | $(CXX) -fcolor-diagnostics -x c - -o /dev/null 2>&1)", "")
  override CFLAGS += -fcolor-diagnostics
endif

# Define colors
NO_COLOR=\033[0m
OK_COLOR=\033[32;01m
ERROR_COLOR=\033[31;01m
WARN_COLOR=\033[33;01m

# Define output strings
OK_STRING=$(OK_COLOR)[OK]$(NO_COLOR)
ERROR_STRING=$(ERROR_COLOR)[ERRORS]$(NO_COLOR)
WARN_STRING=$(WARN_COLOR)[WARNINGS]$(NO_COLOR)

# Commands for processing build output
AWK_CMD = awk '{ printf "Building %-60s -> %-10s\n",$$1, $$2; }'
PRINT_ERROR = printf "$@ $(ERROR_STRING)\n" | $(AWK_CMD) && printf "\n$(CMD)\n\n" && printf %s "$$LOG" && printf "\n" && false
PRINT_WARNING = printf "$@ $(WARN_STRING)\n" | $(AWK_CMD) && printf "\n$(CMD)\n\n" && printf %s "$$LOG" && printf "\n"
PRINT_OK = printf "$@ $(OK_STRING)\n" | $(AWK_CMD)
BUILD_CMD = LOG=$$($(CMD) 2>&1) ; if [ $$? -ne 0 ]; then $(PRINT_ERROR); elif [ "$$LOG" != "" ] ; then $(PRINT_WARNING); else $(PRINT_OK); fi;
#BUILD_CMD = printf "\n$(CMD)"; LOG=$$($(CMD) 2>&1); printf "$(LOG)\n" # USE THIS OPTION TO GET PLAIN OUTPUT FOR DEBUGGING
