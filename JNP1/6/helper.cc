#include "helper.h"

StarshipAttacker::StarshipAttacker(AttackPower attack) : attack(attack)
{
	assert(attack >= 0);
}

StarshipFlyer::StarshipFlyer(Speed speed) : speed(speed)
{
	assert(speed >= 0);
}

StarshipDefender::StarshipDefender(ShieldPoints shield) : shield(shield) {
	assert(shield >= 0);
}

AttackPower StarshipAttacker::getAttackPower() const
{
    return attack;
}

Speed StarshipFlyer::getSpeed() const
{
    return speed;
}

ShieldPoints StarshipDefender::getShield() const
{
    return shield;
}

void StarshipDefender::takeDamage(AttackPower damage)
{
    shield = (damage >= shield) ? 0 : shield - damage;
}

size_t StarshipDefender::getSize() const
{
    return (shield > 0) ? 1 : 0;
}
