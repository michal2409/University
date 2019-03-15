#ifndef REBELFLEET_H_
#define REBELFLEET_H_

#include <memory>

#include "helper.h"

class RebelStarship : public StarshipDefender, public StarshipFlyer
{
 public:
    RebelStarship(ShieldPoints shield, Speed speed);
    virtual AttackPower counterAttack() const;
};

class AttackerRebelStarship : public RebelStarship, public StarshipAttacker
{
 public:
    AttackerRebelStarship(ShieldPoints shield, Speed speed, AttackPower attack);
    virtual AttackPower counterAttack() const override;
};

class Explorer : public RebelStarship
{
 public:
    Explorer(ShieldPoints shield, Speed speed);
};

class StarCruiser : public AttackerRebelStarship
{
 public:
    StarCruiser(ShieldPoints shield, Speed speed, AttackPower attack);
};

class XWing : public AttackerRebelStarship
{
 public:
    XWing(ShieldPoints shield, Speed speed, AttackPower attack);
};

std::shared_ptr<Explorer> createExplorer(ShieldPoints shield, Speed speed);
std::shared_ptr<StarCruiser> createStarCruiser(ShieldPoints shield, Speed speed, AttackPower attack);
std::shared_ptr<XWing> createXWing(ShieldPoints shield, Speed speed, AttackPower attack);

#endif
