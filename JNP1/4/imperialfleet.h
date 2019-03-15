#ifndef IMPERIALFLEETH
#define IMPERIALFLEETH

#include "rebelfleet.h"

template <class U>
class ImperialStarship
{
public:
    typedef U valueType;

    ImperialStarship(U myShield, U myAttackPower) : shield(myShield), attackPower(myAttackPower) {}

    // Zwraca wytrzymałości tarczy.
    U getShield() const
    {
        return shield;
    }

    // Zwraca siłę ataku statku.
    U getAttackPower() const
    {
        return attackPower;
    }

    // Zmniejsza wytrzymałości tarczy o damage, ale nie więcej niż statek ma aktualnie.
    void takeDamage(U damage)
    {
        shield = (damage >= shield) ? static_cast<U>(0) : shield - damage;
    }
private:
    U shield;
    U attackPower;

    // Wykonuje atak na statek Rebelii, obniżając wytrzymałość jego tarczy.
    template<typename R, int minSpeed, int maxSpeed>
    void combatShips(RebelStarship<R, false, minSpeed, maxSpeed> &rebelShip)
    {
        // Atak ma sens gdy oba statkek Imperium jest niezniszczony.
        if (getShield() > static_cast<U>(0))
            rebelShip.takeDamage(getAttackPower());
    }

    // W przypadku gdy R = StarCruiser lub R = XWing (parametr isAttacker == true) atak następuje w dwie strony
    // wytrzymałość tarczy jest obniżana zarówno statkowi Rebelii, jak i statkowi Imperium.
    template<typename R, int minSpeed, int maxSpeed>
    void combatShips(RebelStarship<R, true, minSpeed, maxSpeed>& rebelShip)
    {
        // Atak ma sens gdy oba statki sa niezniszczone.
        if (getShield() > static_cast<U>(0) && rebelShip.getShield() > static_cast<R>(0))
        {
            takeDamage(rebelShip.getAttackPower());
            rebelShip.takeDamage(getAttackPower());
        }
    }

    template<typename I, typename R>
    friend void attack(I &imperialShip, R &rebelShip);
};

// Rozpoczyna atak miedzy statkiem Imperium ze statkiem Rebelii.
template<typename I, typename R>
void attack(I &imperialShip, R &rebelShip)
{
    imperialShip.combatShips(rebelShip);
}

template<class U> using DeathStar = ImperialStarship<U>;
template<class U> using ImperialDestroyer = ImperialStarship<U>;
template<class U> using TIEFighter = ImperialStarship<U>;

#endif