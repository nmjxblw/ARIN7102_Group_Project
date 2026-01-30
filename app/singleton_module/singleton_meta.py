"""单例元类模块"""

# 系统/第三方模块导入
import abc


class SingletonMeta(abc.ABCMeta):
    """
    单例元类，确保一个类只有一个实例。

    使用方法：
        class MyClass(metaclass=SingletonMeta):
            pass

        obj1 = MyClass()
        obj2 = MyClass()
        assert obj1 is obj2  # True
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
