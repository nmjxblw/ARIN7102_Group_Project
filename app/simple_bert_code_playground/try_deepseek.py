from singleton_module.deepseek_manager_new import deepseek_manager_new

manager = deepseek_manager_new
reply = manager.chat(message="解释一下核岭回归")

print(reply)