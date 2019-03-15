#ifndef IMPERIALFLEET_H_
#define IMPERIALFLEET_H_

#include <vector>
#include <initializer_list>
#include <memory>

#include "helper.h"

class ImperialStarship : public StarshipDefender, public StarshipAttacker
{
 public:
    ImperialStarship(ShieldPoints shield, AttackPower attack);
};

class Squadron : public ImperialStarship
{
 public:
    Squadron(std::vector<std::shared_ptr<ImperialStarship>> v);

    virtual void takeDamage(AttackPower damage) override;
    virtual size_t getSize() const override;

 private:
    std::vector<std::shared_ptr<ImperialStarship>> squadVect;
    size_t size = 0;

    void init();
};

class DeathStar : public ImperialStarship
{
 public:
    DeathStar(ShieldPoints myShield, AttackPower myPower);
};

class ImperialDestroyer : public ImperialStarship
{
 public:
    ImperialDestroyer(ShieldPoints myShield, AttackPower myPower);
};

class TIEFighter : public ImperialStarship
{
 public:
    TIEFighter(ShieldPoints myShield, AttackPower myPower);
};

std::shared_ptr<TIEFighter> createTIEFighter(ShieldPoints myShield, AttackPower myPower);
std::shared_ptr<DeathStar> createDeathStar(ShieldPoints myShield, AttackPower myPower);
std::shared_ptr<ImperialDestroyer> createImperialDestroyer(ShieldPoints myShield, AttackPower myPower);
std::shared_ptr<Squadron> createSquadron(std::vector<std::shared_ptr<ImperialStarship>> v);

#endif  // IMPERIALFLEET_H_
