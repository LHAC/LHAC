# There are lots of scripts with cmake
# for finding external libraries. 
# see /usr/local/share/cmake-2.6/Modules/Find*.cmake for more examples
# use cmake -DBLA_VENDOR=openblas to choose OpenBLAS
IF (BLA_VENDOR STREQUAL "openblas")
	find_package(OpenBLAS REQUIRED)
	set (BLAS_LIB ${OpenBLAS_LIB})
ELSE (BLA_VENDOR STREQUAL "openblas")
	find_package(BLAS)
	IF (NOT BLAS_FOUND)
		find_package(OpenBLAS REQUIRED)
		set (BLAS_LIB ${OpenBLAS_LIB})
	ELSE (NOT BLAS_FOUND)
		set (BLAS_LIB ${BLAS_LIBRARIES})
	ENDIF (NOT BLAS_FOUND)
ENDIF (BLA_VENDOR STREQUAL "openblas")

MESSAGE(STATUS "BLAS LIBRARIES: " ${BLAS_LIB})

# add_definitions(-DUSE_CBLAS)
add_executable (lhac drive.cpp Lasso.cpp LogReg.cpp)
target_link_libraries(lhac ${BLAS_LIB}) 