#include <iostream>
#include "strsetconst.h"
#include "strset.h"

namespace {
	#ifdef NDEBUG
		const bool debug = false;
	#else
		const bool debug = true;
	#endif
}

namespace jnp1 {
	unsigned long strset42() {
		static bool first_call = true;
		static unsigned long set_idx;

		if (first_call) {
			first_call = false;
			if (debug)
				std::cerr << "strsetconst init invoked" << std::endl;
			set_idx = jnp1::strset_new();
			jnp1::strset_insert(set_idx, "42");
			if (debug)
				std::cerr << "strsetconst init finished" << std::endl;
		}
		return set_idx;
	}
}