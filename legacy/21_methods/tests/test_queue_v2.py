import pytest
from reliability_v2.queue_d import choose_recipe, dino_extend, validate_device_indices, idle_device


def history(last,previous=None,n=30):
    values=[last-.1]*(n-10)+[last if previous is None else previous]*5+[last]*5
    return [{'epoch':i+1,'dev':{'macro_auc':v}} for i,v in enumerate(values)]


def test_recipe_choice_source_last5_two_seed_average_and_tie():
    candidates=[{'lr':3e-5,'epochs':30,'histories':[history(.901),history(.899)]},
      {'lr':1e-5,'epochs':30,'histories':[history(.8998),history(.9)]},
      {'lr':5e-5,'epochs':30,'histories':[history(.88),history(.99)],'qualification':'FAIL'}]
    assert choose_recipe(candidates)['lr']==1e-5


def test_dino_extension_any_seed_adjacent_windows_and_cap():
    assert dino_extend([history(.8,.797),history(.8,.8)],30)==45
    assert dino_extend([history(.8,.797,n=45)],45)==60
    assert dino_extend([history(.8,.7,n=60)],60) is None
    assert dino_extend([history(.8,.799)],30) is None


def test_protected_devices_always_rejected():
    assert validate_device_indices([0,1,2,3])==[0,1,2,3]
    with pytest.raises(ValueError):validate_device_indices([4,5,6,7])
    with pytest.raises(ValueError):validate_device_indices([0,0])


def test_zero_memory_existing_compute_process_is_not_idle():
    assert idle_device({'uuid':'gpu0','used_mib':0},set())
    assert not idle_device({'uuid':'gpu0','used_mib':0},{'gpu0'})
