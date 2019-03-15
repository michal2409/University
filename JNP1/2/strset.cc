#include <iostream>
#include <map>
#include <set>
#include <limits>
#include <cassert>
#include <cstring>
#include "strset.h"
#include "strsetconst.h"

namespace {
	unsigned long id = 0;
	using string = std::string;
	using map_of_sets = std::map<unsigned long, std::set<string>>;

	map_of_sets& sets_map() {
		static auto m = new map_of_sets();
		return *m;
	}

	unsigned long get_set_42_idx() {
		static unsigned long idx;
		static bool first_call = true;
		if (first_call) {
			first_call = false;
			idx = jnp1::strset42();
		}
		return idx;
	}

	#ifdef NDEBUG
		const bool debug = false;
	#else
		const bool debug = true;
	#endif

	void log_entrace(const string &func_name, const string &msg) {
		std::cerr << func_name << "(" << msg << ")" << std::endl;
	}

	void log_lack_id_set(const string &func_name, unsigned long id) {
		std::cerr << func_name << ": set " << id << " does not exist" << std::endl;
	}

	void print_set_type(unsigned long id) {
		if (id == get_set_42_idx())
			std::cerr << "the 42 Set";
		else
			std::cerr << "set " << id;
	}

	void log_action(const string &func_name, unsigned long id, const string &msg) {
		std::cerr << func_name << ": ";
		print_set_type(id);
		std::cerr << msg << std::endl;
	}

	void log_action(const string &func_name, unsigned long id, const string &msg1, const string &value, const string &msg2) {
		std::cerr << func_name << ": ";
		print_set_type(id);
		std::cerr << msg1 << value << msg2 << std::endl;
	}

	void log_invalid_set_42_action(const string &func_name, const string msg) {
		std::cerr << func_name << ": attempt to " << msg << " the 42 Set" << std::endl;
	}

	bool invalid_param(const string &func_name, unsigned long id, const char* value) {
		if (value != nullptr)
			return false;
		if (debug) {
			log_entrace(func_name, std::to_string(id) + ", NULL");
			std::cerr << func_name << ": invalid value (NULL)" << std::endl;
		}
		return true;
	}
}

namespace jnp1 {
	unsigned long strset_new() {
		if (debug) {
			assert(id < std::numeric_limits<unsigned int>::max()); // Sprawdzenie czy nastapil overflow unsigned longa.
			log_entrace(__func__, "");
		}

		sets_map().emplace(id, std::set<string>());
		if (debug)
			std::cerr << __func__ << ": set " << id << " created" << std::endl;

		return id++;
	}

	void strset_delete(unsigned long id) {
		if (debug)
			log_entrace(__func__, std::to_string(id));

		if (id == get_set_42_idx()) { // Proba usuniecia zbioru 42.
			if (debug)
				log_invalid_set_42_action(__func__, "remove");
			return;
		}

		auto it = sets_map().find(id);

		if (it == sets_map().end()) { // Brak id zbioru.
			if (debug)
				log_lack_id_set(__func__, id);
			return;
		}

		it->second.clear();
		sets_map().erase(it);
		if (debug)
			log_action(__func__, id, " deleted");
	}

	size_t strset_size(unsigned long id) {
		if (debug)
			log_entrace(__func__, std::to_string(id));

		auto it = sets_map().find(id);
		size_t set_size = 0;

		if (it == sets_map().end()) { // Brak id zbioru.
			if (debug)
				log_lack_id_set(__func__, id);
			return set_size;
		}

		set_size = sets_map()[id].size();
		if (debug)
			log_action(__func__, id, " contains " + std::to_string(set_size) + " element(s)");

		return set_size;
	}

	void strset_insert(unsigned long id, const char* value) {
		if (invalid_param(__func__, id, value))
			return;

		if (debug)
			log_entrace(__func__, std::to_string(id) + ", \"" + value + "\"");

		if (id == get_set_42_idx() && strcmp(value, "42")) {
			if (debug)
				log_invalid_set_42_action(__func__, "insert into");
			return;
		}

		auto it = sets_map().find(id);
		if (it == sets_map().end()) { // Brak id zbioru.
			if (debug)
				log_lack_id_set(__func__, id);
			return;
		}

		auto ret = it->second.emplace(value);
		if (!ret.second) {
			if (debug && id == get_set_42_idx() && !strcmp(value, "42"))
				log_invalid_set_42_action(__func__, "insert into");
			else if (debug)
				log_action(__func__, id, ", element \"", value, "\" was already present");
			return;
		}

		if (debug)
			std::cerr << __func__ << ": set " << id << ", element \"" << value << "\" inserted" << std::endl;
	}

	void strset_remove(unsigned long id, const char* value) {
		if (invalid_param(__func__, id, value))
			return;

		if (debug)
			log_entrace(__func__, std::to_string(id) + ", \"" + value + "\"");

		if (id == get_set_42_idx()) { // Proba usuniecia ze zbioru 42.
			if (debug)
				log_invalid_set_42_action(__func__, "remove from");
			return;
		}

		auto it = sets_map().find(id);
		if (it == sets_map().end()) { // Brak id zbioru.
			if (debug)
				log_lack_id_set(__func__, id);
			return;
		}

		auto it_set = it->second.find(value);
		if (it_set == it->second.end()) { // Brak elementu value.
			if (debug)
				log_action(__func__, id, " does not contain the element \"", value, "\"");
			return;
		}

		it->second.erase(it_set);
		if (debug)
			log_action(__func__, id, ", element \"", value, "\" removed");
	}

	int strset_test(unsigned long id, const char* value) {
		if (invalid_param(__func__, id, value))
			return 0;

		if (debug)
			log_entrace(__func__, std::to_string(id) + ", \"" + value + "\"");

		auto it = sets_map().find(id);
		if (it == sets_map().end()) { // Brak id zbioru.
			if (debug)
				log_lack_id_set(__func__, id);
			return 0;
		}

		if (it->second.find(value) == it->second.end()) { // Brak elementu value.
			if (debug)
				log_action(__func__, id, " does not contain the element \"", value, "\"");
			return 0;
		}

		if (debug)
			log_action(__func__, id, " contains the element \"", value, "\"");

		return 1;
	}

	void strset_clear(unsigned long id) {
		if (debug)
			log_entrace(__func__, std::to_string(id));

		if (id == get_set_42_idx()) { // Proba usuniecia elementow zbioru 42.
			if (debug)
				log_invalid_set_42_action(__func__, "clear");
			return;
		}

		auto it = sets_map().find(id);
		if (it == sets_map().end()) { // Brak id zbioru.
			if (debug)
				log_lack_id_set(__func__, id);
			return;
		}

		it->second.clear();
		if (debug)
			log_action(__func__, id, " cleared");
	}

	int strset_comp(unsigned long id1, unsigned long id2) {
		if (debug)
			log_entrace(__func__, std::to_string(id1) + ", " + std::to_string(id2));

		auto id1_it = sets_map().find(id1), id2_it = sets_map().find(id2);
		std::set<string> set1 = (id1_it != sets_map().end() ? id1_it->second : std::set<string>());
		std::set<string> set2 = (id2_it != sets_map().end() ? id2_it->second : std::set<string>());

		int res = -1;
		bool finished = false;
		auto it1 = set1.begin(), it2 = set2.begin();

		while (it1 != set1.end() && it2 != set2.end()) {
			int compare_result = (*it1).compare(*it2);
			if (compare_result > 0) {
				res = 1;
				finished = true;
				break;
			}
			if (compare_result < 0) {
				res = -1;
				finished = true;
				break;
			}
			it1++; it2++;
		}

		if (!finished) {
			if (it1 == set1.end() && it2 == set2.end())
				res = 0;
			else if (it1 != set1.end())
				res = 1;
			else
				res = -1;
		}

		if (debug) {
			std::cerr << __func__ << ": result of comparing ";
			print_set_type(id1);
			std::cerr << " to ";
			print_set_type(id2);
			std::cerr << " is " << res << std::endl;

			if (id1_it == sets_map().end())
				log_lack_id_set(__func__, id1);
			if (id2_it == sets_map().end())
				log_lack_id_set(__func__, id2);
		}

		return res;
	}
}
