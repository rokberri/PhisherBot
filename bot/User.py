class User:
    def __init__(self, user_id, user_name, user_class) -> None:
        self.__user_id = user_id
        self.__user_name = user_name
        self.__user_role = user_class

    @property
    def user_role(self):
        return self.__user_role

    @user_role.setter
    def user_type(self, value):
        self.__user_role = value

    def __del__(self):
        pass
        
    def print_info(self) -> None:
        print(f'User Id: {self.__user_id}')
        print(f'User Name: {self.__user_name}')
        print(f'User Class: {self.__user_role}')