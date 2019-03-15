#ifndef BATTLEH
#define BATTLEH

#include <iostream>
#include <algorithm>
#include <tuple>
#include <array>
#include "imperialfleet.h"
#include "rebelfleet.h"

template <typename T, T t0, T t1, typename... S>
class SpaceBattle
{
public:
	SpaceBattle(S... s) : time(t0), rebelsCount(0), imperialCount(0), ships(s...)
	{
		if constexpr (sizeof...(S) > 0)
			countShips<0>(std::get<0>(ships));
	}

	// Zwraca liczbę niezniszczonych statków Imperium.
	size_t countImperialFleet()
	{
		return imperialCount;
	}

	// Zwraca liczbę niezniszczonych statków Rebelii.
	size_t countRebelFleet()
	{
		return rebelsCount;
	}

	// Na początku sprawdza stan statkow, jesli bitwa sie zakonczyla to wypisuje komunikat o rezultacie
	// w przeciwnym razie sprawdza aktualny czas; jeśli jest to czas ataku,
	// to następuje atak statków Imperium na statki Rebelii; na koniec czas przesuwa się o timeStep.
	void tick(T timeStep)
	{
		if (imperialCount == 0 && rebelsCount == 0)
			std::cout << "DRAW\n";
		else if (imperialCount == 0)
			std::cout << "REBELLION WON\n";
		else if (rebelsCount == 0)
			std::cout << "IMPERIUM WON\n";
		else
		{
			if (std::binary_search(squares.begin(), squares.end(), time))
			{
				if constexpr (sizeof...(S) > 0)
					battle<0>(std::get<0>(ships));
			}
		}
		time = (time + timeStep) % (t1 + static_cast<T>(1));
	}

private:
	static_assert(t0 <= t1 && static_cast<T>(0) <= t0);

	T time;
	size_t rebelsCount;
	size_t imperialCount;
	std::tuple<S...> ships;

	// Oblicza liczbe kwadratów w przedziale [0, t1]
	static constexpr size_t numberOfSquares()
	{
		size_t x = static_cast<size_t>(t1) + 1;
		if (x == 0 || x == 1)
			return x;

		size_t start = 1, end = x, ans = 0;
		while (start <= end)
		{
			size_t mid = (start + end) / 2;
			if (mid * mid == x)
				return mid;
			if (mid * mid < x)
			{
				start = mid + 1;
				ans = mid;
			}
			else
				end = mid - 1;
		}
		return ans + 1;
	}

	// Wypelnia tablice kwadratami z przedzialu [0, t1]
	static constexpr auto generateSquares()
	{
		constexpr size_t size = numberOfSquares();
		std::array<size_t, size> squares = {};
		for (size_t i = 0; i < size; ++i)
			squares[i] = i * i;
		return squares;
	}

	static constexpr auto squares = generateSquares();

	// Aktualizuje liczbe statkow imperium.
	template<size_t i, typename W>
	void countShips(const ImperialStarship<W> &ship)
	{
		if (ship.getShield() > static_cast<W>(0))
			imperialCount++;
		if constexpr (i + 1 < sizeof...(S))
			countShips<i + 1>(std::get<i + 1>(ships));
	}

	// Aktualizuje liczbe statkow rebelii.
	template<size_t i, typename W, bool isAttacker, int minSpeed, int maxSpeed>
	void countShips(const RebelStarship<W, isAttacker, minSpeed, maxSpeed> &ship)
	{
		if (ship.getShield() > static_cast<W>(0))
			rebelsCount++;
		if constexpr (i + 1 < sizeof...(S))
			countShips<i + 1>(std::get<i + 1>(ships));
	}

	// Statek imperium nie atakuje drugiego staku imperium, szuka dalej statku rebelii.
	template<size_t i, typename W, typename V>
	void attackRebel(ImperialStarship<W> &ship, ImperialStarship<V> &attacker)
	{
		(void)ship; // uciszenie warningu
		if constexpr (i + 1 < sizeof...(S))
			attackRebel<i + 1>(std::get<i + 1>(ships), attacker);
	}

	// Statek imperium atakuje statek rebelii.
	template<size_t i, typename W, bool isAttacker, int minSpeed, int maxSpeed,typename V>
	void attackRebel(RebelStarship<W, isAttacker, minSpeed, maxSpeed> &ship, ImperialStarship<V> &attacker)
	{
		if (ship.getShield() > static_cast<W>(0))
		{
			attack(attacker, ship);
			if(attacker.getShield() == static_cast<V>(0))
				imperialCount--;
			if(ship.getShield() == static_cast<W>(0))
				rebelsCount--;
		}
		if(attacker.getShield() > static_cast<V>(0))
		{
			if constexpr (i + 1 < sizeof...(S))
				attackRebel<i + 1>(std::get<i + 1>(ships), attacker);
		}
	}

	// Statek imperium atakuje wszystkie mozliwe statki rebelii.
	template<size_t i, typename V>
	void battle(ImperialStarship<V> &ship)
	{
		if (ship.getShield() > static_cast<V>(0))
			attackRebel<0>(std::get<0>(ships), ship);
		if constexpr (i + 1 < sizeof...(S))
			battle<i + 1>(std::get<i + 1>(ships));
	}

	// Statek rebelii nie atakuje, wywoluje funkcje dla nastepnego statku.
	template<size_t i, typename V, bool isAttacker, int minSpeed, int maxSpeed>
	void battle(RebelStarship<V, isAttacker, minSpeed, maxSpeed> &ship)
	{
		(void)ship; // uciszenie warningu
		if constexpr (i + 1 < sizeof...(S))
			battle<i + 1>(std::get<i + 1>(ships));
	}
};

#endif