#ifndef REBELFLEETH
#define REBELFLEETH

#include <cassert>

template <class U, bool isAttacker, int minSpeed, int maxSpeed>
class RebelStarship
{
public:
    typedef U valueType;

    RebelStarship(U myShield, U mySpeed) : shield(myShield), speed(mySpeed)
    {
        static_assert(!isAttacker);
        assert(checkSpeed());
    }

    RebelStarship(U myShield, U mySpeed, U myAttackPower) : shield(myShield),speed(mySpeed), attackPower(myAttackPower)
    {
        static_assert(isAttacker);
        assert(checkSpeed());
    }

    // Zwraca wytrzymałości tarczy.
    U getShield() const
    {
        return shield;
    }

    // Zwraca prędkość statku.
    U getSpeed() const
    {
        return speed;
    }

    // Zmniejsza wytrzymałości tarczy o damage, ale nie więcej niż statek ma aktualnie.
    void takeDamage(U damage)
    {
        shield = (damage > shield) ? static_cast<U>(0) : shield - damage;
    }

    // Zwraca siłę ataku statku.
    U getAttackPower() const
    {
         static_assert(isAttacker);
         return attackPower;
    }

private:
    U shield;
    U speed;
    U attackPower;

    bool checkSpeed()
    {
        return speed >= static_cast<U>(minSpeed) && speed <= static_cast<U>(maxSpeed);
    }
};

template<class U> using Explorer = RebelStarship<U, false, 299796, 2997960>;
template<class U> using StarCruiser = RebelStarship<U, true, 99999, 299795>;
template<class U> using XWing = RebelStarship<U, true, 299796, 2997960>;

#endif