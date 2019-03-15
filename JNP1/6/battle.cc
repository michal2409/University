#include "battle.h"

Strategy1::Strategy1(Time startTime, Time maxTime)
    : currTime(startTime)
    , maxTime(maxTime)
{
    assert(0 <= startTime && startTime <= maxTime);
}

bool Strategy1::isAttackTime()
{
    return ((currTime % 2 == 0 || currTime % 3 == 0) && currTime % 5 != 0);
}

void Strategy1::tickTime(Time timeStep)
{
    currTime = (currTime + timeStep) % (maxTime + 1);
}

SpaceBattle::SpaceBattle(Time startTime, Time maxTime, rebelVect rebels,
        imperialVect imperials, size_t rebelsCount, size_t imperialCount)
    : rebels(std::move(rebels))
    , imperials(std::move(imperials))
    , rebelsCount(rebelsCount)
    , imperialCount(imperialCount)
    , strategy(std::unique_ptr<Strategy>(new Strategy1(startTime, maxTime)))
{}

size_t SpaceBattle::countImperialFleet() const
{
    return imperialCount;
}

size_t SpaceBattle::countRebelFleet() const
{
    return rebelsCount;
}

void SpaceBattle::tick(Time timeStep)
{
    if (imperialCount == 0 && rebelsCount == 0)
        std::cout << "DRAW" << std::endl;
    else if (imperialCount == 0)
        std::cout << "REBELLION WON" << std::endl;
    else if (rebelsCount == 0)
        std::cout << "IMPERIUM WON" << std::endl;
    else
        if (strategy->isAttackTime())
            battle();

    strategy->tickTime(timeStep);
}

void SpaceBattle::battle() {
    for (auto imp = imperials.begin(); imp != imperials.end(); ++imp)
    {
        for (auto reb = rebels.begin(); reb != rebels.end(); ++reb)
        {
            if (imp->get()->getShield() > 0 && reb->get()->getShield() > 0)
            {
                size_t imperialsBeforeAttack = imp->get()->getSize();

                reb->get()->takeDamage(imp->get()->getAttackPower());
                imp->get()->takeDamage(reb->get()->counterAttack());

                imperialCount -= imperialsBeforeAttack - imp->get()->getSize();
                rebelsCount -= 1 - reb->get()->getSize();
            }
        }
    }
}

SpaceBattle::Builder& SpaceBattle::Builder::startTime(Time t)
{
   currTime = t;
   return *this;
}

SpaceBattle::Builder& SpaceBattle::Builder::maxTime(Time t)
{
    _maxTime = t;
    return *this;
}

SpaceBattle::Builder& SpaceBattle::Builder::ship(rebelPtr rebel)
{
    rebels.push_back(rebel);
    rebelsCount += rebel->getSize();
    return *this;
}

SpaceBattle::Builder& SpaceBattle::Builder::ship(imperialPtr imperial)
{
    imperials.push_back(imperial);
    imperialCount += imperial->getSize();
    return *this;
}

SpaceBattle SpaceBattle::Builder::build()
{
    return SpaceBattle{currTime, _maxTime, rebels, imperials, rebelsCount, imperialCount};
}
