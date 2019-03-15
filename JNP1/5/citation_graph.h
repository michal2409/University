#ifndef CITATION_GRAPH_H_
#define CITATION_GRAPH_H_

#include <vector>
#include <memory>
#include <map>
#include <set>

class PublicationNotFound : public std::exception {
	const char* what() const noexcept {
		return "PublicationNotFound";
	}
};

class PublicationAlreadyCreated : public std::exception {
	const char* what() const noexcept {
		return "PublicationAlreadyCreated";
	}
};

class TriedToRemoveRoot : public std::exception {
	const char* what() const noexcept {
		return "TriedToRemoveRoot";
	}
};

template <class Publication>
class CitationGraph {
 private:
	struct Node;
	using id_type = typename Publication::id_type;;
	using nodes_map = std::map<id_type, std::weak_ptr<Node>>;

	struct Node {
		Publication publication;
		std::set<std::shared_ptr<Node>> children;
		std::set<std::weak_ptr<Node>, std::owner_less<std::weak_ptr<Node>>> parents;
		typename nodes_map::iterator it;
		std::weak_ptr<nodes_map> graph_nodes;

		Node(id_type const &stem_id, std::shared_ptr<nodes_map> graph_nodes)
							: publication(stem_id), graph_nodes(graph_nodes) {
			it = graph_nodes->end();
		}

		~Node() {
			auto graph_nodes_ptr = graph_nodes.lock();
			if (it != graph_nodes_ptr->end()) {
				for (auto &child: children) {
					child->parents.erase(it->second);
				}
				graph_nodes_ptr->erase(it);
			}
		}
	};

	std::shared_ptr<nodes_map> nodes;
	std::shared_ptr<Node> root;

	std::shared_ptr<Node> get_node(const id_type &id) const {
		auto node_it = nodes->find(id);
		if (node_it == nodes->end()) {
			throw PublicationNotFound();
		}
		return node_it->second.lock();
	}

 public:
	CitationGraph(id_type const &stem_id) {
		nodes = std::make_shared<nodes_map>();
		root = std::make_shared<Node>(stem_id, nodes);
		root->it = nodes->emplace(stem_id, root).first;
	}

	CitationGraph(CitationGraph<Publication> &&other) noexcept : nodes(std::move(other.nodes)),
	                                                             root(std::move(other.root)){}

	CitationGraph<Publication>& operator=(CitationGraph<Publication> &&other) noexcept {
		if (this != &other) {
			nodes.swap(other.nodes);
			root.swap(other.root);
		}
		return *this;
	}

	CitationGraph(CitationGraph<Publication> const &other) = delete;
	CitationGraph& operator=(CitationGraph<Publication> const &other) = delete;

	id_type get_root_id() const noexcept(noexcept(std::declval<Publication>().get_id())) {
		return root.get()->publication.get_id();
	}

	bool exists(const id_type &id) const {
		return nodes->find(id) != nodes->end();
	}

	std::vector<id_type> get_children(const id_type &id) const {
		auto node = get_node(id);
		std::vector<id_type> children_ids;
		for (auto &child : node->children) {
			children_ids.push_back(child->publication.get_id());
		}
		return children_ids;
	}

	std::vector<id_type> get_parents(const id_type &id) const {
		auto node = get_node(id);
		std::vector<id_type> parents_ids;
		for (auto &parent : node->parents) {
			parents_ids.push_back(parent.lock()->publication.get_id());
		}
		return parents_ids;
	}

	Publication& operator[](const id_type &id) const {
		return get_node(id)->publication;
	}

	void create(id_type const &id, id_type const &parent_id) {
		create(id, std::vector<id_type >{parent_id});
	}

	void create(id_type const &id, std::vector<id_type> const &parent_ids) {
		if (parent_ids.empty()) {
			throw PublicationNotFound();
		}
		for (auto parent : parent_ids) {
			if (!exists(parent)) {
				throw PublicationNotFound();
			}
		}
		if (exists(id)) {
			throw PublicationAlreadyCreated();
		}

		auto node_to_add = std::make_shared<Node>(id, nodes);
		std::vector<std::shared_ptr<Node>> updated_parents;
		try {
			for (auto &parent_id : parent_ids) {
				auto parent = nodes->find(parent_id)->second.lock();
				updated_parents.push_back(parent);
				parent->children.insert(node_to_add);
				node_to_add->parents.insert(parent);
			}
			node_to_add->it = nodes->emplace(id, node_to_add).first;
		}
		catch (...) {
			for (auto &parent: updated_parents) {
				parent->children.erase(node_to_add);
				node_to_add->parents.erase(parent);
			}
			throw;
		}
	}

	void add_citation(id_type const &child_id, id_type const &parent_id) {
		auto child_node = get_node(child_id);
		auto parent_node = get_node(parent_id);
		child_node->parents.insert(parent_node);
		try {
			parent_node->children.insert(child_node);
		}
		catch(...) {
			child_node->parents.erase(parent_node);
			throw;
		}
	}

	void remove(id_type const &id) {
		if (root->publication.get_id() == id) {
			throw TriedToRemoveRoot();
		}
		auto node_to_remove = get_node(id);
		for (auto &parent: node_to_remove->parents) {
			parent.lock()->children.erase(node_to_remove);
		}
	}
};

#endif // CITATION_GRAPH_H_