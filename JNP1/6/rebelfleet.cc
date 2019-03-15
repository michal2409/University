#include "rebelfleet.h"

RebelStarship::RebelStarship(ShieldPoints shield, Speed speed)
    : StarshipDefender(shield), StarshipFlyer(speed)
{}

AttackerRebelStarship::AttackerRebelStarship(ShieldPoints shield, Speed speed, AttackPower attack)
    : RebelStarship(shield, speed)
    , StarshipAttacker(attack)
{}

Explorer::Explorer(ShieldPoints shield, Speed speed)
    : RebelStarship(shield, speed)
{
    assert(299796 <= speed && speed <= 2997960);
}

StarCruiser::StarCruiser(ShieldPoints shield, Speed speed, AttackPower attack)
    : AttackerRebelStarship(shield, speed, attack)
{
    assert(99999 <= speed && speed <= 299795);
}

XWing::XWing(ShieldPoints shield, Speed speed, AttackPower attack)
    : AttackerRebelStarship(shield, speed, attack)
{
    assert(299796 <= speed && speed <= 2997960);
}

AttackPower RebelStarship::counterAttack() const
{
    return 0;
}

AttackPower AttackerRebelStarship::counterAttack() const
{
    return getAttackPower();
}

std::shared_ptr<Explorer> createExplorer(ShieldPoints shield, Speed speed)
{
    return std::make_shared<Explorer>(shield, speed);
}

std::shared_ptr<StarCruiser> createStarCruiser(ShieldPoints shield, Speed speed, AttackPower attack)
{
    return std::make_shared<StarCruiser>(shield, speed, attack);
}

std::shared_ptr<XWing> createXWing(ShieldPoints shield, Speed speed, AttackPower attack)
{
    return std::make_shared<XWing>(shield, speed, attack);
}
