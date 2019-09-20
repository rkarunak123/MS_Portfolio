/*-----Create Table Scripts---------*/

Drop table if exists Student
Drop table if exists Employee_Location
Drop table if exists Class_Schedule
Drop table if exists Class_Location
Drop table if exists Class
Drop table if exists Event_Location_Schedule
Drop table if exists Studio_Location
Drop table if exists Parent
Drop table if exists Employee
Drop table if exists Employee_Type
Drop table if exists [Event]
Drop table Adult_Student_Info

--Table : Employee_Type
Create table Employee_Type(
Employee_Type_ID int identity not null,
Employee_Type varchar(20) not null,
Constraint Employee_Type_PK Primary Key(Employee_Type_ID),
Constraint Employee_Type_U1 Unique (Employee_Type)
)
GO

--Table : Employee
Create table Employee(
Employee_ID int identity not null,
Employee_First_Name varchar(20) not null,
Employee_Last_Name varchar(20) not null,
SSN varchar(9) not null,
DOB Date not null,
Street_Address varchar(15) not null,
City varchar(15) not null,
State varchar(15) not null default 'California',
ZipCode varchar(5) not null,
Salary numeric(9,2) not null,
Employee_Type_ID int not null,
Constraint Employee_PK Primary Key(Employee_ID),
Constraint Employee_FK1 Foreign Key(Employee_Type_ID) references  Employee_Type(Employee_Type_ID)
)
GO

--Table Sudio_Location
Create table Studio_Location(
Studio_Location_ID int identity not null,
Location_Name varchar(30) not null,
Location_Addresss varchar(15) not null,
City varchar(15) not null,
State varchar(15) not null,
ZipCode varchar(5) not null,
Start_Year char(4) not null,
Manager_ID int not null,
Constraint Studio_Location_PK Primary Key (Studio_Location_ID),
Constraint Studio_Location_FK Foreign Key (Manager_ID) references Employee(Employee_ID)
)
GO

--Table : Employee_Location
Create table Employee_Location(
Employee_Location_ID int identity not null,
Employee_ID int not null,
Studio_Location_ID int not null,
Constraint Employee_Location_PK Primary Key (Employee_Location_ID),
Constraint Employee_Location_FK1 Foreign Key (Employee_ID) references Employee(Employee_ID),
Constraint Employee_Location_FK2 Foreign Key (Studio_Location_ID) references Studio_Location(Studio_Location_ID)
)
GO

--Table : Class
Create table Class(
Class_ID int identity not null,
Class_Name varchar(20) not null,
Class_Description varchar(30) not null,
Max_Class_Size int not null,
Constraint Class_PK Primary Key (Class_ID),
Constraint Class_U1 Unique(Class_Name)
) 
GO

--Table Class_Location
Create table Class_Location(
Class_Location_ID int identity not null,
Class_ID int not null,
Studio_Location_ID int not null,
Constraint Class_Location_PK Primary Key (Class_Location_ID),
Constraint Class_Location_FK1 Foreign Key (Class_ID) references Class(Class_ID),
Constraint Class_Location_FK2 Foreign Key (Studio_Location_ID) references Studio_Location(Studio_Location_ID)
)
GO

--Table Parent
Create table Parent(
Parent_ID int identity not null,
Parent_First_Name varchar(30) not null,
Parent_Last_Name varchar(30) not null,
Street_Address varchar(15) not null,
City varchar(15) not null,
State varchar(15) not null,
ZipCode varchar(5) not null,
Email varchar(50) not null,
Phone varchar(15) not null,
Constraint Parent_PK Primary Key (Parent_ID)
)
GO

--Table Class_Schedule
Create table Class_Schedule(
Class_Schedule_ID int identity not null,
Class_Location_ID int not null,
Instructor_ID int not null,
Class_Start_Time varchar(6) not null,
Class_End_Time varchar(6) not null,
Day_Of_The_Week varchar(10) not null,
Constraint Class_Schedule_PK Primary Key (Class_Schedule_ID),
Constraint Class_Schedule_FK1 Foreign Key (Class_Location_ID) references Class_Location(Class_Location_ID),
Constraint Class_Schedule_FK2 Foreign Key (Instructor_ID) references Employee(Employee_ID)
)
GO

--Table Student
Create table Student(
Student_ID int identity not null,
Student_First_Name varchar(30) not null,
Student_Last_Name varchar(30) not null,
DOB Date not null,
Parent_ID int, -- I made this field optional since adult students may not have a parent
Class_Schedule_ID int not null,
Constraint Student_PK Primary Key (Student_ID),
Constraint Student_FK1 Foreign Key (Parent_ID) references Parent(Parent_ID),
Constraint Student_FK2 Foreign Key (Class_Schedule_ID) references Class_Schedule(Class_Schedule_ID)
)
GO

--Table Event
Create table [Event](
Event_ID int identity not null,
Event_Name varchar(20) not null,
Event_Description varchar(30) ,
Constraint Event_PK primary key (Event_ID),
Constraint Event_U1 unique (Event_Name)
)
GO

--Create table Event_Location_Schedule
Create table Event_Location_Schedule(
Event_Location_Schedule_ID int identity not null,
Studio_Location_ID int not null,
Event_ID int not null,
Event_Date DateTime not null,
Constraint Event_Location_Schedule_PK Primary Key (Event_Location_Schedule_ID),
Constraint Event_Location_Schedule_FK1 Foreign Key(Studio_Location_ID) references Studio_Location(Studio_Location_ID),
Constraint Event_Location_Schedule_FK2 Foreign Key(Event_ID) references [Event](Event_ID)
)
GO

--Create table Adult_Student_Info
Create table Adult_Student_Info(
Adult_Student_Info_ID int identity not null,
Street_Address varchar(15) not null,
City varchar(15) not null,
State varchar(15) not null,
ZipCode varchar(5) not null,
Email varchar(50) not null,
Phone varchar(15) not null,
Constraint Adult_Student_Info_PK Primary Key (Adult_Student_Info_ID)
)
GO

/*-----End Create Table Scripts---------*/

/*---- Needed to make some corrections to the tables---------------*/

--Typo in Column Name Location_Address
SP_RENAME 'Studio_Location.Location_Addresss', 'Location_Address', 'COLUMN'
GO

-- Increasing the of some column since they could hold longer string values
Alter table Employee
Alter Column Street_Address varchar(30) not null
GO

Alter table Studio_Location
Alter Column Location_Address varchar(30) not null
GO

Alter table Parent
Alter Column Street_Address varchar(30) not null
GO

--Decided to Add AgeGroup column to table Class
Alter table Class
Add  Age_Group varchar(15) not null
GO

Alter table Class
Alter Column  Class_Description varchar(50) not null
GO

Alter table [Event]
Alter Column Event_Name varchar(30) not null
GO

 Alter table [Event]
 Alter Column Event_Description varchar(50)
GO

--Added FK to table Student for table Adult_Student_Info
Alter table Student
Add  Adult_Student_Info_ID int -- I made this an optional since only Adult Sudents will have this field populate
GO

Alter table Student
Add  Constraint Student_FK3 Foreign Key (Adult_Student_Info_ID) references Adult_Student_Info(Adult_Student_Info_ID)
GO

/*----End  Needed to make some corrections to the tables---------------*/


/*----Insert Record into tables ------*/

--Table Employee_Type
Insert into Employee_Type(Employee_Type)
values('FullTime'),('PartTime'),('Apprentice')
GO

--Review to see if data is correct
Select * from Employee_Type
GO


--Table Employee

--I'm not including the State Column in the insert because I want it to default to 'California'
Insert into Employee(Employee_First_Name, Employee_Last_Name, SSN, DOB, Street_Address, City, ZipCode, Salary, Employee_Type_ID)
values('Daniel','Martinez','600345999','03/04/2000','34 Cross Street','San Jose','97654',40000,1)

--The rest of the inserts have State Column specified
Insert into Employee(Employee_First_Name, Employee_Last_Name, SSN, DOB, Street_Address, City, State, ZipCode, Salary, Employee_Type_ID)
values('Gina','Daniel','603853856','09/21/2003','3435 Mulberry Ct','San Jose','California','93411',10000,3)

Insert into Employee(Employee_First_Name, Employee_Last_Name, SSN, DOB, Street_Address, City, ZipCode, Salary, Employee_Type_ID)
values('Kim','Lee','938647846','01/20/1995','456 HorseShoe Lane','San Jose','94633',60000,1)

Insert into Employee(Employee_First_Name, Employee_Last_Name, SSN, DOB, Street_Address, City, ZipCode, Salary, Employee_Type_ID)
values('Sam','Victor','653647888','07/22/1999','BlueCross Road','Santa Clara','97364',60000,2)

Insert into Employee(Employee_First_Name, Employee_Last_Name, SSN, DOB, Street_Address, City, ZipCode, Salary, Employee_Type_ID)
values('Shirley','Lou','65367833','03/04/2001','First Street','Sunnyvale','95633',60000,2)

--Review to see if data is correct
Select * from Employee
GO

-- Table Studio Location
Insert into Studio_Location(Location_Name,Location_Address,City,State, ZipCode,Start_Year,Manager_ID)
values('America''s Best Karate','12 street','San Jose','California','93532','2015',3)

Insert into Studio_Location(Location_Name,Location_Address,City,State, ZipCode,Start_Year,Manager_ID)
values('Victory Marital Arts','3444 Mckee Avenue','Santa Clara','California','93266','2010',3)

Insert into Studio_Location(Location_Name,Location_Address,City,State, ZipCode,Start_Year,Manager_ID)
values('Xtremen Martial Arts','78 Walnut Grove','Sunnyvale','California','95623','2017',4)

--Review to see if data is correct
Select * from Studio_Location
GO

--Delete from Studio_Location since I accidently inserted duplicate rows
Delete from Studio_Location where Studio_Location_ID between 1 and 7
GO

--Table Class
Insert into Class(Class_Name,Class_Description,Max_Class_Size,Age_Group)
values('Freshman','Beginner''s Class Belts - white to yellow1 belt',30, '5 and above')

Insert into Class(Class_Name,Class_Description,Max_Class_Size,Age_Group)
values('Sophomore','Intermediate 1 - orange to green1',30, '5 and above') 

Insert into Class(Class_Name,Class_Description,Max_Class_Size,Age_Group)
values('Junior','Intermediate 2 - purple to blue1',30, '5 and above')

Insert into Class(Class_Name,Class_Description,Max_Class_Size,Age_Group)
values('Senior','Advanced  - brown to red-black',30, '5 and above')

Insert into Class(Class_Name,Class_Description,Max_Class_Size,Age_Group)
values('Graduates','Proficient - black belt',20, '10 and above')

Insert into Class(Class_Name,Class_Description,Max_Class_Size,Age_Group)
values('Adult','Cardio Class',20, '14 and above')

Insert into Class(Class_Name,Class_Description,Max_Class_Size,Age_Group)
values('Pee-Wee','Toddlers',20, '3 and above')

--Review to see if data is correct
Select * from Class
GO

--Table Class_Location
Insert into Class_Location(Class_ID,Studio_Location_ID)
values(1,8),(1,10),(2,8),(2,9),(2,10),(3,9),(3,10) ,(4,8),(5,9),(6,10),(6,8),(7,8) ,(5,8),(2,8),(3,10)

--Review to see if data is correct
select * from Class_Location

--Remove Duplicates
delete from Class_Location
where Class_Location_ID=22

delete from Class_Location
where Class_Location_ID=23

--Review to see if data is correct
select * from Class_Location

GO

--Table Class_Schedule
Insert into Class_Schedule(Class_Location_ID,Instructor_ID,Class_Start_Time,Class_End_Time,Day_Of_The_Week)
values(9,1,'5:30pm','6:30pm','Monday')

Insert into Class_Schedule(Class_Location_ID,Instructor_ID,Class_Start_Time,Class_End_Time,Day_Of_The_Week)
values(9,1,'4:30pm','6:00pm','Friday')

Insert into Class_Schedule(Class_Location_ID,Instructor_ID,Class_Start_Time,Class_End_Time,Day_Of_The_Week)
values(10,2,'5:00pm','6:00pm','Wednesday')

Insert into Class_Schedule(Class_Location_ID,Instructor_ID,Class_Start_Time,Class_End_Time,Day_Of_The_Week)
values(11,2,'2:30pm','3:30pm','Saturday')

Insert into Class_Schedule(Class_Location_ID,Instructor_ID,Class_Start_Time,Class_End_Time,Day_Of_The_Week)
values(11,3,'5:30pm','6:30pm','Monday')

Insert into Class_Schedule(Class_Location_ID,Instructor_ID,Class_Start_Time,Class_End_Time,Day_Of_The_Week)
values(12,4,'4:30pm','5:30pm','Tuesday')

Insert into Class_Schedule(Class_Location_ID,Instructor_ID,Class_Start_Time,Class_End_Time,Day_Of_The_Week)
values(13,5,'5:30pm','6:30pm','Friday')

Insert into Class_Schedule(Class_Location_ID,Instructor_ID,Class_Start_Time,Class_End_Time,Day_Of_The_Week)
values(12,5,'1:30pm','3:30pm','Saturday')

Insert into Class_Schedule(Class_Location_ID,Instructor_ID,Class_Start_Time,Class_End_Time,Day_Of_The_Week)
values(14,2,'5:30pm','6:30pm','Monday')

Insert into Class_Schedule(Class_Location_ID,Instructor_ID,Class_Start_Time,Class_End_Time,Day_Of_The_Week)
values(15,1,'7:00pm','8:30pm','Tuesday')

Insert into Class_Schedule(Class_Location_ID,Instructor_ID,Class_Start_Time,Class_End_Time,Day_Of_The_Week)
values(16,3,'7:00pm','8:30pm','Thursday')


Insert into Class_Schedule(Class_Location_ID,Instructor_ID,Class_Start_Time,Class_End_Time,Day_Of_The_Week)
values(19,3,'5:00pm','6:30pm','Saturday')

--Review to see if data is correct
Select * from Class_Schedule
GO

Update Class_Schedule
set Class_Start_Time = '5:30pm'
where Class_Schedule_ID = 24

Update Class_Schedule
set Class_Start_Time = '5:00pm'
where Class_Schedule_ID = 23

Select * from Class_Schedule
GO

--Table Parent
Insert into Parent(Parent_First_Name,Parent_Last_Name,Street_Address, City, State, ZipCode,Email,Phone)
values('Rebecca','Karunakaran','264 Joshua Court','San Jose','California','97656','Rebecca_Karunakaran@gmail.com','408-865-9776')

Insert into Parent(Parent_First_Name,Parent_Last_Name,Street_Address, City, State, ZipCode,Email,Phone)
values('James','Smith','6 Washington Way','Santa Clara','California','97656','James_Smith@yahoo.com','408-463-5424')

Insert into Parent(Parent_First_Name,Parent_Last_Name,Street_Address, City, State, ZipCode,Email,Phone)
values('Gina','Parker','78 North Street','San Jose','California','95432','Gina_Parker@gmail.com','765-443-5235')

Insert into Parent(Parent_First_Name,Parent_Last_Name,Street_Address, City, State, ZipCode,Email,Phone)
values('John','Franklin','336 Main Street','Sunnyvale','California','94322','John_Franklin@hotmail.com','408-459-1112')

--Review to see if data is correct
Select * from Parent
GO


--Table Student
Insert into Student(Student_First_Name,Student_Last_Name,DOB,Parent_ID,Class_Schedule_ID)
values('Liam','Smith','01/20/2010',2,12)


Insert into Student(Student_First_Name,Student_Last_Name,DOB,Parent_ID,Class_Schedule_ID)
values('Ava','Smith','01/20/2012',2,13)


Insert into Student(Student_First_Name,Student_Last_Name,DOB,Parent_ID,Class_Schedule_ID)
values('Nia','Karunakaran','06/06/2011',1,19)


Insert into Student(Student_First_Name,Student_Last_Name,DOB,Parent_ID,Class_Schedule_ID)
values('Kevin','Parker','09/11/2008',3,32)


Insert into Student(Student_First_Name,Student_Last_Name,DOB,Parent_ID,Class_Schedule_ID)
values('Joseph','Frankin','01/20/2007',4,28)

--Inserted an Adult Records who will not have a Parent Record
Insert into Student(Student_First_Name,Student_Last_Name,DOB,Parent_ID,Class_Schedule_ID)
values('Zion','Charles','01/20/1998',NULL,32)

Insert into Student(Student_First_Name,Student_Last_Name,DOB,Parent_ID,Class_Schedule_ID)
values('Gina','Davis','01/20/1985',NULL,32)


--Review to see if data is correct
select *  from Student
GO


--Updating Student Zion Charles to an Adult class since he is over 18years
Update Student
set Class_Schedule_ID = 34
where Student_ID = 11
GO

--Updating Student Gina Davis to an Adult class since she is over 18years
Update Student
set Class_Schedule_ID = 34
where Student_ID = 12
GO


--Added new table Adult_Student_Info and need to update values in the Student table for the Adult record 
Update Student
Set Adult_Student_Info_ID = 1
where Student_ID = 11
GO

Update Student
Set Adult_Student_Info_ID = 2,
where Student_ID = 12
GO

--Review to see if data is correct
select *  from Student
GO


--Table Event
Insert into [Event](Event_Name,Event_Description)
values('Black Belt Ceremony','Black Belt will be awraded to qualifying students'),
('Ice-Cream Social',null), ('Movie Night',null)

--Review to see if data is correct
Select * from [Event]
GO

--Table Event_Location_Schedule
Insert into Event_Location_Schedule(Studio_Location_ID,Event_ID,Event_Date)
values(8,1,'06/09/2018 5:00PM'), (8,2,'09/09/2018 6:30PM'), (9,1,'07/05/2018 5:30PM'), (10,3,'08/23/2018 7:00 PM'), (10,1,'06/20/2018 4:00PM')
GO

--Review to see if data is correct
Select * from  Event_Location_Schedule
GO

--Table Employee_Location
Insert into Employee_Location(Employee_ID,Studio_Location_ID)
values(1,8),(1,10),(2,9),(3,8),(4,8),(5,10),(3,9)
GO

--Review to see if data is correct
Select * from  Employee_Location
GO

--Table Adult_Student_Info
Insert into Adult_Student_Info(Street_Address,City,State,ZipCode,Email,Phone)
values('56 Blue Hawk Dr','Morgan Hill','California','95436','Zion_Charles@yahoo.com','408-938-8473')
GO

Insert into Adult_Student_Info(Street_Address,City,State,ZipCode,Email,Phone)
values('600 Shorelane','Sunnyvale','California','92323','Gine_Davis@hotmail.com','408-364-2222')
GO

--Review to see if data is correct
Select *  from  Adult_Student_Info
GO


/*-----------End Insert Statements------------------*/



select * from Student left join Parent on Parent.Parent_ID = Student.Parent_ID join Class_Schedule c on c.Class_Schedule_ID=Student.Class_Schedule_ID
join Class_Location cl on cl.Class_Location_ID = c.Class_Location_ID join Class on Class.Class_ID=cl.Class_ID


--------------View-----------------
------View to track Studio Information -----------
drop View StudioInfoView
GO

Create View StudioInfoView As
Select Studio_Location.Location_Name,
Studio_Location.Location_Address as StreetAddress ,
Studio_Location.City, 
Studio_Location.State , 
Studio_Location.ZipCode, 
(Employee.Employee_First_Name + ' ' + Employee.Employee_Last_Name) as Manager,
Count(Employee.Employee_ID) as TotalEmployees -- Used Aggregate Function Count to find total employee for a studio
From Studio_Location 
Join Employee on Studio_Location.Manager_ID = Employee.Employee_ID
join Employee_Location on Employee_Location.Studio_Location_ID = Studio_Location.Studio_Location_ID
Group by Studio_Location.Location_Name,
Studio_Location.Location_Address ,
Studio_Location.City, 
Studio_Location.State , 
Studio_Location.ZipCode, 
(Employee.Employee_First_Name + ' ' + Employee.Employee_Last_Name)
GO


Select * from StudioInfoView
GO

-----View to track Class Schedule for each of their studio locations --------
Drop View ClassScheduleStudioLocationView
GO

Create View ClassScheduleStudioLocationView as 
Select Studio_Location.Location_Name as Studio,
Class.Class_Name as Class,
(Class_Schedule.Class_Start_Time + ' to ' + Class_Schedule.Class_End_Time) as ClassTiming,
Class_Schedule.Day_Of_the_Week as Day,
(Employee.Employee_First_Name + ' ' + Employee.Employee_Last_Name) as Instructor
from Class_Schedule 
join Class_Location on Class_Schedule.Class_Location_ID = Class_Location.Class_Location_ID
join Class on Class_Location.Class_ID = Class.Class_ID
join Studio_Location on Class_Location.Studio_Location_ID = Studio_Location.Studio_Location_ID
join Employee on Class_Schedule.Instructor_ID = Employee.Employee_ID
GO

Select * from ClassScheduleStudioLocationView
GO

-----View to track Event Schedule for each of their studio locations --------
Drop View EventScheduleStudioLocationView
GO

Create View EventScheduleStudioLocationView as 
Select Studio_Location.Location_Name as Studio,
Event.Event_Name as Event,
(FORMAT(Event_Location_Schedule.Event_Date,'dd/MM/yy') + '  ' + FORMAT(Event_Location_Schedule.Event_Date,'hh:mm tt')) as Date -- want to convert into readable format with AM/PM
from Event_Location_Schedule 
join Event on Event_Location_Schedule.Event_ID = Event.Event_ID
join Studio_Location on Event_Location_Schedule.Studio_Location_ID = Studio_Location.Studio_Location_ID
GO

--View  Events for all Studios
Select * from EventScheduleStudioLocationView
order by Studio
GO
 
 --View Event per Studio 
Select * from EventScheduleStudioLocationView
where Studio = 'Victory Marital Arts'
GO
 


 
-----View to track Student Information --------
Drop View StudentInformationView
GO

Create View StudentInformationView as 
Select(Student.Student_First_Name + ' ' + Student.Student_Last_Name) as StudentName,
isnull((Parent.Parent_First_Name + ' ' + Parent.Parent_Last_Name),'N/A') as ParentName, --Adult Students will not have a Parent Record
convert(varchar, Student.DOB, 101) as DOB,
(DATEDIFF (yy, Student.DOB, GETDATE())) as Age,
(Case When Student.Parent_ID is not null then  Parent.Phone else Adult_Student_Info.Phone end) as  ContactNumber,
 Studio_Location.Location_Name as Studio,
Class.Class_Name as Class ,
Class_Schedule.Day_Of_The_Week as Day,
(Class_Schedule.Class_Start_Time + ' to ' + Class_Schedule.Class_End_Time) as ClassTiming
from Student
left join Parent on Student.Parent_ID = Parent.Parent_ID -- left join to get Student records that have not Parent Records because they are Adults
join Class_Schedule on Student.Class_Schedule_ID = Class_Schedule.Class_Schedule_ID
join Class_Location on  Class_Schedule.Class_Location_ID = Class_Location.Class_Location_ID
join Studio_Location on Class_Location.Studio_Location_ID = Studio_Location.Studio_Location_ID
left join Adult_Student_Info on Student.Adult_Student_Info_ID =Adult_Student_Info.Adult_Student_Info_ID
join Class on Class_Location.Class_ID =  Class.Class_ID
GO

--View  Students for all Studios
Select * from StudentInformationView
order by Studio,Class
GO
 
 --View Student per Studio 
Select * from StudentInformationView
where Studio = 'Victory Marital Arts'
GO
 
 -----View to track Employee Information --------
Drop View EmployeeInformationView
GO


Create View EmployeeInformationView as 
Select (Employee.Employee_First_Name + ' ' + Employee.Employee_Last_Name) as EmployeeName,
Studio_Location.Location_Name as Studio,
Class.Class_Name,
(Class_Schedule.Class_Start_Time + ' to ' + Class_Schedule.Class_End_Time) as ClassTiming,
Class_Schedule.Day_Of_The_Week Day
from Employee
join Class_Schedule on Class_Schedule.Instructor_ID = Employee.Employee_ID
join Class_Location on Class_Schedule.Class_Location_ID  = Class_Location.Class_Location_ID
join Class on Class_Location.Class_ID = Class.Class_ID
join Studio_Location on Class_Location.Studio_Location_ID = Studio_Location.Studio_Location_ID 



select * from EmployeeInformationView
order by EmployeeName,Studio
GO 

select * from EmployeeInformationView
where EmployeeName = 'Sam Victor'
GO 

------Stored Procedures and Functions -------------

--Function to retrieve the Employee_Type_ID for a given Employee_Type
Create Function dbo.Lookup_EmployeeType_ID(@Employee_Type varchar(20))
Returns int as 
Begin
	Declare @returnValue int 
	
	Select @returnValue =  Employee_Type_ID 
	from Employee_Type
	Where Employee_Type =  @Employee_Type

	--Send the Employee_Type_ID back 
	return @returnValue
End
Go

--Check to see if function works correctly

Select *, dbo.Lookup_EmployeeType_ID(Employee_Type) as ReturnValueFromFN
from Employee_Type


---Stored Procedure to Add/Modify Employee Information
Drop Procedure dbo.sp_AddModify_EMployee_Info

Create Procedure sp_AddModify_Employee_Info(
                                               @Employee_First_Name varchar(20),
											   @Employee_Last_Name varchar(20),
											   @Employee_SSN varchar(9),
											   @Employee_DOB Date,
											   @Street_Address varchar(30),
											   @City varchar(15),
											   @State varchar(15),
											   @ZipCode varchar(5),
											   @Salary numeric(9,2),
											   @Employee_Type varchar(20)
											   )
AS
Begin
	
	Declare @Employee_Type_ID int

   --Get the Employee_Type_ID fisrt for the Employee_Type provided as the input parameter
   Select @Employee_Type_ID = dbo.Lookup_EmployeeType_ID(@Employee_Type) 
   From Employee_Type

	--Check to see if the record exist. If it does, update the record otherwise add the record
	If Exists (Select * from Employee where Employee_First_Name = @Employee_First_Name and Employee_Last_Name = @Employee_Last_Name)
		Begin
			Update Employee
			Set SSN = @Employee_SSN,
				DOB = @Employee_DOB,
				Street_Address = @Street_Address,
				City = @City,
				State = @State,						  
				ZipCode	= @ZipCode,						   
				Salary = @Salary,							   
				Employee_Type_id = dbo.Lookup_EmployeeType_ID(@Employee_Type) -- Using the lookup function to get FK Employee_Type_ID
			Where Employee_First_Name = @Employee_First_Name
			And Employee_Last_Name = @Employee_Last_Name

	    End
	Else
	   Begin
			Insert into Employee(Employee_First_Name, Employee_Last_Name, SSN, DOB, Street_Address, City, State, ZipCode, Salary, Employee_Type_ID)
            values(@Employee_First_Name, @Employee_Last_Name, @Employee_SSN, @Employee_DOB, @Street_Address, @City, @State, @ZipCode, @Salary, @Employee_Type_ID)
			Return @@IDENTITY -- If this has a value, then I know we inserted the record
		End
End	
							  
Go

delete from EMployee where Employee_ID = 7

--Insert a new record
Declare @ReturnValue int
EXEC @ReturnValue =  dbo.sp_AddModify_Employee_Info @Employee_First_Name = 'Liam',
								   @Employee_Last_Name = 'Mathew', 
								   @Employee_SSN = '6993673333', 
								   @Employee_DOB = '08/20/1995', 
								   @Street_Address = '56 Main Stree',
								   @City = 'San Jose',
								   @State = 'California', 
								   @ZipCode = '93456', 
								   @Salary = 40000, 
								   @Employee_Type = 'PartTime'
			

SELECT * FROM Employee where Employee_ID = @ReturnValue
GO

select * from EMployee

--Update an the same record, changing city from 'San Jose' to 'Sunnyvale'
Declare @ReturnValue int
EXEC @ReturnValue =  dbo.sp_AddModify_Employee_Info @Employee_First_Name = 'Liam',
								   @Employee_Last_Name = 'Mathew', 
								   @Employee_SSN = '6993673333', 
								   @Employee_DOB = '08/20/1995', 
								   @Street_Address = '56 Main Stree',
								   @City = 'Sunnyvale',--'San Jose' to 'Sunnyvale
								   @State = 'California', 
								   @ZipCode = '93456', 
								   @Salary = 40000, 
								   @Employee_Type = 'PartTime'
			
GO

select * from EMployee

-------------------------------------------

--Function to retrieve the Class_Schedule_ID based on the input parameters 
Drop Function dbo.Lookup_Class_Schedule_ID

Create Function dbo.Lookup_Class_Schedule_ID(@Class varchar(20),@Studio varchar(30),@Day_Of_Week varchar(10),@Class_Start_Time varchar(6))
Returns int as 
Begin
	Declare @returnValue int , @Class_Location_ID int, @Class_ID int, @Studio_Location_ID int
	
	--Get the Class_ID and Studio_Location_ID
	Select @Class_ID = Class_ID from Class where Class_Name = @Class
	Select @Studio_Location_ID = Studio_Location_ID from Studio_Location where Location_Name = @Studio

	--Get the Class_Location_ID first
	Select @Class_Location_ID  = Class_Location_ID
	from Class_Location
	Where Class_ID = @Class_ID   
	And   Studio_Location_ID = @Studio_Location_ID 

	Select @returnValue =  Class_Schedule_ID 
	from Class_Schedule
	Where Class_Location_ID = @Class_Location_ID
	and Day_of_the_Week = @Day_Of_Week
	and Class_Start_Time = @Class_Start_Time

	--Send the  Class_Schedule_ID  back 
	return   @returnValue
End
Go


--Check to see if function works correctly

Select Class.Class_Name, Studio_Location.Location_Name,Class_Schedule.Class_Schedule_ID,Class_Schedule.Class_Location_ID,
dbo.Lookup_Class_Schedule_ID(Class.Class_Name, Studio_Location.Location_Name,Class_Schedule.Day_Of_The_Week,Class_Schedule.Class_Start_Time) as ReturnValueFromFunction
from Class_Schedule
 join Class_Location on  Class_Schedule.Class_Location_ID = Class_Location.Class_Location_ID
 join Class on Class.Class_ID = Class_Location.Class_ID
 join Studio_Location on Class_Location.Studio_Location_ID = Studio_Location.Studio_Location_ID
order by Class_Schedule_ID desc 

--Function to check if the class is full 
Drop Function dbo.Is_Class_Full

Create Function dbo.Is_Class_Full(@Class varchar(20),@Studio varchar(30),@Day_Of_Week varchar(10),@Class_Start_Time varchar(6))
Returns int as 
Begin

	Declare @returnValue int , @Class_Size int , @Class_Count int
	
	--Get the Class Size 
	Select @Class_Size = Max_Class_Size from Class where Class_Name = @Class
	
	-- Get the Count of Student for the given class/studio/dayofweek/starttime combination
	Select @Class_Count =  count(Student_ID)
	from Student
	inner join Class_Schedule on Student.Class_Schedule_ID = Class_Schedule.Class_Schedule_ID
	join Class_Location on Class_Location.Class_Location_ID = dbo.Lookup_Class_Schedule_ID(@Class, @Studio,@Day_Of_Week,@Class_Start_Time)

	--Send the return value which indicates whether class is full or no
	If (@Class_Count < @Class_Size )
	  Select @returnValue = 0 --Class is not full
	else
	 Select @returnValue = 1 --Class is full

	return @returnValue
  
End
GO

--Check to see if function works correctly


Select Class.Class_Name, Studio_Location.Location_Name,Class_Schedule.Class_Start_Time, Class_Schedule.Day_Of_The_Week,count(Student_ID) as Count,
dbo.Is_Class_Full(Class.Class_Name, Studio_Location.Location_Name,Class_Schedule.Day_Of_The_Week,Class_Schedule.Class_Start_Time) as ReturnValueFromFunction
from Student
join Class_Schedule on Student.Class_Schedule_ID = Class_Schedule.Class_Schedule_ID
 join Class_Location on  Class_Schedule.Class_Location_ID = Class_Location.Class_Location_ID
 join Class on Class.Class_ID = Class_Location.Class_ID
 join Studio_Location on Class_Location.Studio_Location_ID = Studio_Location.Studio_Location_ID
 group by Class.Class_Name, Studio_Location.Location_Name,Class_Schedule.Class_Start_Time, Class_Schedule.Day_Of_The_Week
order by Class_Name, Studio_Location.Location_Name



---Stored Procedure to Add/Modify Student Information
Drop Procedure dbo.sp_AddModify_Student_Info

Create Procedure sp_AddModify_Student_Info(
                                               @Student_First_Name varchar(20),
											   @Student_Last_Name varchar(20),
											   @Student_DOB Date,
											   @Parent_First_Name varchar(20),
											   @Parent_Last_Name varchar(20),
											   @Street_Address varchar(30),
											   @City varchar(15),
											   @State varchar(15),
											   @ZipCode varchar(5),
											   @Email varchar(50),
											   @Phone varchar(15),
											   @Class varchar(20),
											   @Studio varchar(30),
											   @Day_Of_Week varchar(10),
											   @Class_Start_Time varchar(6)
											   )
As
Begin
			Declare @IsAdult char(1), @Class_Schedule_ID int, @Parent_ID int, @Adult_Student_Info int, @Is_Class_Full int
			
			--Check to see if the Student is an Adult or not, this will determine which to add the contact information
            If (datediff(yy,@Student_DOB,getdate()) >= 18) 
			    select @IsAdult=1 
			Else 
				     select @IsAdult=0 
			

			--Get the Class_Schedule_ID based on the input parameters 
			Select @Class_Schedule_ID = dbo.Lookup_Class_Schedule_ID(@Class,@Studio,@Day_Of_Week,@Class_Start_Time) 
			From Class_Schedule


			--Check to see if the record exist. If it does, update the record otherwise add the record
			If Exists (Select * from Student
			           where Student_First_Name = @Student_First_Name 
					   and Student_Last_Name = @Student_Last_Name)
			Begin 
			        --Record Exists so Modify
					  Update Student
					  Set DOB = @Student_DOB,
						  Class_Schedule_ID = @Class_Schedule_ID
					  Where Student_First_Name = @Student_First_Name 
					  And Student_Last_Name = @Student_Last_Name	


					  If (@IsAdult=0) 
						Begin
				          --If Student is a Minor, the address is stored in the Parent table	 
							Update Parent
							Set  Street_Address = @Street_Address,
								 City = @City,
					             State = @State,						  
							     ZipCode	= @ZipCode,						   
							     Email  = @Email, 	
							    Phone  = @Phone
							Where Parent_ID = (Select Parent_ID from Student where 	Student_First_Name = @Student_First_Name 
					                   And Student_Last_Name = @Student_Last_Name)
						 End
					    Else
						 Begin
				        -- Student is an Adult, the address is stored in Adult_Student_Info table
							Update Adult_Student_Info
							Set  Street_Address = @Street_Address,
							     City = @City,
					             State = @State,						  
							     ZipCode	= @ZipCode,						   
							     Email  = @Email, 	
							     Phone  = @Phone
							Where Adult_Student_Info_ID = (Select Adult_Student_Info_ID from Student where 	Student_First_Name = @Student_First_Name 
					                   And Student_Last_Name = @Student_Last_Name)

									   
					
						 End
			End
			Else
			Begin
			  --Record does not exists so Add
              -- But first see if the class is full or not
			  Select @Is_Class_Full =  dbo.Is_Class_Full(@Class, @Studio,@Day_Of_Week,@Class_Start_Time)

			 If (@Is_Class_Full = 0) -- which means it not full 
			 Begin
				If (@IsAdult=0) 
						Begin
						     --Student is a minor, so add the Parent Record first
							 Insert into Parent(Parent_First_Name,Parent_Last_Name,Street_Address, City, State, ZipCode,Email,Phone)
                             values(@Parent_First_Name,@Parent_Last_Name,@Street_Address, @City, @State, @ZipCode,@Email,@Phone)

							 -- we should get the Parent_ID from the newly added record 
							 Select @Parent_ID = @@IDENTITY 

							 --Finally insert the student record
							 Insert into Student(Student_First_Name,Student_Last_Name,DOB,Parent_ID, Class_Schedule_ID)
							 values (@Student_First_Name, @Student_Last_Name, @Student_DOB, @Parent_ID,@Class_Schedule_ID)
						End
				Else
				        Begin
						     --Student is an adult, so add a Adult_Student_Info Record first
							 Insert into Adult_Student_Info(Street_Address,City,State,ZipCode,Email,Phone)
							 values(@Street_Address, @City, @State, @ZipCode, @Email, @Phone)

							 -- we should get the Adult_Student_Info_ID from the newly added record 
							 Select @Adult_Student_Info = @@IDENTITY 

							 --Finally insert the student record
							 Insert into Student(Student_First_Name,Student_Last_Name,DOB,Class_Schedule_ID, Adult_Student_Info_ID)
							 values (@Student_First_Name, @Student_Last_Name, @Student_DOB,@Class_Schedule_ID , @Adult_Student_Info)
						End
				End
			End
End
GO


--Update existing student who is a minor to change the DOB from '01/02/2012' to '03/05/2012
-- and Steet Address from '6 Washington Way' to '600 Washington Way'
select * from Student
select * from Parent
select * from Class_Schedule where Class_Schedule_ID = 13

--Get the Class Schedule Info for 'Ava Smith'
select Class_Schedule.class_schedule_ID,Class.Class_Name,Studio_Location.Location_Name , Class_Schedule.Day_Of_The_Week, Class_Schedule.Class_Start_Time
from Student
join Class_Schedule  on Student.Class_Schedule_ID = Class_Schedule.Class_Schedule_ID
join Class_Location on Class_Schedule.Class_Location_ID = Class_Location.Class_Location_ID 
join Class on Class_Location.Class_ID = Class.Class_ID
join Studio_Location on Studio_Location.Studio_Location_ID = Class_Location.Studio_Location_ID
where Student.Student_First_Name = 'Ava' and Student.Student_Last_Name = 'Smith'

--Now Update the record
EXEC   dbo.sp_AddModify_Student_Info  'Ava','Smith','03/05/2012','James','Smith','600 Washington Way',
'Santa Clara','California','97656','James_Smith@yahoo.com','408-463-5424',
'Freshman','Victory Marital Arts','Friday','4:30pm'


Select * from Student where Student_First_Name = 'Ava' And Student_Last_Name = 'Smith' 

Select * from Parent  where Parent_ID = (select Parent_ID from Student 
where Student_First_Name = 'Ava' And Student_Last_Name = 'Smith' )

--Update existing student 'Zion Charles' who is an Adult to change the DOB from '01/20/1998' to '01/25/1998'
-- and Email from 'Zion_Charles@yahoo.com' to 'Zion_Charles@hotmail.com'

select * from Student where Student_First_Name ='Zion' and  Student_Last_Name = 'Charles'

select * from Adult_Student_Info where Adult_Student_Info_ID = 1

--Get the Class Schedule Info for 'Zion Charles'
select Class_Schedule.class_schedule_ID,Class.Class_Name,Studio_Location.Location_Name , Class_Schedule.Day_Of_The_Week, Class_Schedule.Class_Start_Time
from Student
join Class_Schedule  on Student.Class_Schedule_ID = Class_Schedule.Class_Schedule_ID
join Class_Location on Class_Schedule.Class_Location_ID = Class_Location.Class_Location_ID 
join Class on Class_Location.Class_ID = Class.Class_ID
join Studio_Location on Studio_Location.Studio_Location_ID = Class_Location.Studio_Location_ID
where Student.Student_First_Name = 'Zion' and Student.Student_Last_Name = 'Charles'

--Execute the Stored Procedure to  Update the record
Declare @ReturnValue int
Exec @ReturnValue =   dbo.sp_AddModify_Student_Info  'Zion','Charles','01/25/1998',NULL,NULL,'56 Blue Hawk Dr',
'Morgan Hill','California','95436','Zion_Charles@hotmail.com','408-938-8473',
'Adult','Victory Marital Arts','Saturday','5:00pm'
select @ReturnValue

select * from Student where Student_First_Name ='Zion' and  Student_Last_Name = 'Charles'


select * from Adult_Student_Info where Adult_Student_Info_ID = 1

---Let's use the Stored Proc to Add a new student who is a minor

--Execute the Stored Procedure to  Update the record
EXEC   dbo.sp_AddModify_Student_Info   dbo.sp_AddModify_Student_Info  'Taylor','Swift','01/25/2003','Ronald','Smith','34 Hollywood Dr',
'San Jose','California','95452','Ronald_Smith@gmail.com','408-256-2436',
'Adult','Victory Marital Arts','Saturday','5:00pm'


select * from Student 

select * from Parent


---Let's use the Stored Proc to Add a new student who is an Adult

--Execute the Stored Procedure to  Update the record
EXEC   dbo.sp_AddModify_Student_Info     'Tom','James','01/25/1990',NULL,NULL,'34 Eagle Lane',
'San Jose','California','93432','Tom_James@yahoo.com','125-343-6564',
'Junior','America''s Best Karate','Tuesday','7:00pm'

select * from Student 

select * from Adult_Student_Info

---Let's use the Stored Proc to Add a new student who is an Adult

--Execute the Stored Procedure to  Update the record
EXEC   dbo.sp_AddModify_Student_Info     'Morgan','Nick','06/20/1991',NULL,NULL,'34 Eagle Lane',
'San Jose','California','93432','Mogan_Nick@yahoo.com','125-343-6564',
'Junior','America''s Best Karate','Tuesday','7:00pm'

select * from Student 

select * from Adult_Student_Info


---------------------------------------------------------------------------
select dbo.Lookup_Class_Schedule_ID('Freshman','Victory Marital Arts','Friday','4:30pm')

--Get the Class_ID and Studio_Location_ID
	Select  Class_ID from Class where Class_Name = 'Freshman'
	Select  Studio_Location_ID from Studio_Location where Location_Name = 'Victory Marital Arts'

Select  Class_Location_ID
	from Class_Location
	Where Class_ID = 1 
	And   Studio_Location_ID = 8

	Select  Class_Schedule_ID 
	from Class_Schedule
	Where Class_Location_ID = 9
	and Day_of_the_Week = 'Friday'
	and Class_Start_Time = '4:30pm'

select * from Class_Schedule --12,9
select * from Class_Location where Class_Location_ID in (11,22) 11  9 --1,8
select * from Class where Class_ID = 2 1 -- Freshma
Select * from Studio_Location where Studio_Location_ID = 8 --Vic

Select * --@returnValue =  Class_Schedule_ID 
	from Class_Schedule
	order by Class_Location_ID, Day_Of_The_Week

select Class_Location_ID,Class_ID,Studio_Location_ID 
from Class_Location   order by class_ID, Studio_Location_ID


select * from Class_Schedule where Class_Location_ID in (11,22,15,23)

select * from Class_Location

select * from employee

 Update Student
					  Set DOB = '03/05/2012',
						  Class_Schedule_ID = 13
					  Where Student_First_Name = 'Ava' 
					  And Student_Last_Name = 'Smith'

select top 100 * from rcensus