#include "imperialfleet.h"

ImperialStarship::ImperialStarship(ShieldPoints shield, AttackPower attack)
    : StarshipDefender(shield)
    , StarshipAttacker(attack)
{}

Squadron::Squadron(std::vector<std::shared_ptr<ImperialStarship>> v)
    : ImperialStarship(0, 0)
    , squadVect(std::move(v))
    , size(0)
{
    init();
}

void Squadron::takeDamage(AttackPower damage)
{
    for (auto it = squadVect.begin(); it != squadVect.end(); ++it)
    {
        if ((it->get()->getShield() == 0))
            continue;

        auto sizeBeforeAttack = it->get()->getSize();
        auto shieldBeforeAttack = it->get()->getShield();
        auto attackBeforeAttack = it->get()->getAttackPower();
        it->get()->takeDamage(damage);

        if (it->get()->getShield() == 0)
        {
            shield -= shieldBeforeAttack;
            attack -= attackBeforeAttack;
        }
        else
        {
            shield -= damage;
            attack -= attackBeforeAttack - it->get()->getAttackPower();
        }

        size -= sizeBeforeAttack - it->get()->getSize();
    }
}

size_t Squadron::getSize() const
{
    return size;
}

void Squadron::init()
{
    for (auto it = squadVect.begin(); it != squadVect.end(); ++it)
    {
        if ((it->get()->getShield() == 0))
            continue;

        shield += it->get()->getShield();
        attack += it->get()->getAttackPower();
        size += it->get()->getSize();
    }
}

DeathStar::DeathStar(ShieldPoints myShield, AttackPower myPower)
    : ImperialStarship(myShield, myPower)
{}

ImperialDestroyer::ImperialDestroyer(ShieldPoints myShield, AttackPower myPower)
    : ImperialStarship(myShield, myPower)
{}

TIEFighter::TIEFighter(ShieldPoints myShield, AttackPower myPower)
    : ImperialStarship(myShield, myPower)
{}

std::shared_ptr<TIEFighter> createTIEFighter(ShieldPoints myShield, AttackPower myPower)
{
    return std::make_shared<TIEFighter>(myShield, myPower);
}

std::shared_ptr<DeathStar> createDeathStar(ShieldPoints myShield, AttackPower myPower)
{
    return std::make_shared<DeathStar>(myShield, myPower);
}

std::shared_ptr<ImperialDestroyer> createImperialDestroyer(ShieldPoints myShield, AttackPower myPower)
{
    return std::make_shared<ImperialDestroyer>(myShield, myPower);
}

std::shared_ptr<Squadron> createSquadron(std::vector<std::shared_ptr<ImperialStarship>> v)
{
    return std::make_shared<Squadron>(std::move(v));
}
