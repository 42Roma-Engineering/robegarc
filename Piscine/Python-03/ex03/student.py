class Person:

    def __init__(self, name, last_name) :
        self.name = name
        self.last_name = last_name

    def __str__(self):
        return f"{self.name} {self.last_name}"


class Student(Person):

    def __init__(self, name, last_name, is_student=False) :
        super().__init__(name, last_name)
        self.is_student = is_student
        self.degree_course = ""

    def __str__(self):
        if self.is_student and self.degree_course :
            return f"{self.name} {self.last_name} is registered to {self.degree_course}"
        else:
            return f"{self.name} {self.last_name} is not registered to any course"


def main():

    name = input("Insert first name: ")
    surname = input("Insert last name: ")
    is_student = input("Are you a student? (y/n): ").lower()

    for i in range(0, 3) :
        if is_student != "y" and is_student != "n" :
            is_student = input("Please type \"y\" or \"n\": ").lower()
        else :
            break

    person = Person(name, surname)

    if is_student == "y":
        degree_course = input("Enter your degree course: ")
        student = Student(name, surname, is_student=True)
        student.degree_course = degree_course
        print(student)
    else:
        print(person)


if __name__ == "__main__":
    main()

