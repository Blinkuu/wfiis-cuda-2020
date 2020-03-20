#include "get_type_name.h"

template<>
std::string get_type_name<int>() {
	return "int";
}

template<>
std::string get_type_name<float>() {
	return "float";
}
