global global_var1
global_var1 = "Original"

def initialize():        
    print("Accessing in initialize section:",global_var1)
    global_var1 = "initialize"
    return

def fun1():
    print("Accessing in fun1:",global_var1)
    return
def fun2():
    print("Accessing in fun2:",global_var1)
    return

def main():
    print("Accessing in main - start:",global_var1)
    initialize()
    fun1()
    fun2()
    print("Accessing in main - final:",global_var1)
    return

if __name__ == "__main__":
    main()
