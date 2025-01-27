import pyirk as p

__URI__ = "irk:/olaa/0.1"
keymanager = p.KeyManager()
p.register_mod(__URI__, keymanager)
p.start_mod(__URI__)

# add content here

p.end_mod()
