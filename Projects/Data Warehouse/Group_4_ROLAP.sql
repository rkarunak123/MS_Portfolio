/****** Object:  Database ist722_hhkhan_od4_dw    Script Date: 5/26/2019 4:06:23 PM ******/

/*
Group 4: Rebecca Karunakaran, Haley Brown, Kyle Wojtaszek
*/

/*
Kimball Group, The Microsoft Data Warehouse Toolkit
Generate a database from the datamodel worksheet, version: 4
You can use this Excel workbook as a data modeling tool during the logical design phase of your project.
As discussed in the book, it is in some ways preferable to a real data modeling tool during the inital design.
We expect you to move away from this spreadsheet and into a real modeling tool during the physical design phase.
The authors provide this macro so that the spreadsheet isn't a dead-end. You can 'import' into your
data modeling tool by generating a database using this script, then reverse-engineering that database into
your tool.
Uncomment the next lines if you want to drop and create the database
*/
USE ist722_hhkhan_od4_dw
--GO
--CREATE SCHEMA project
--GO

-- DROP Fact tables first to untangle foreign keys

/* Drop table project.FactConsolidatedSalesReporting */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'project.FactConsolidatedSalesReporting') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE project.FactConsolidatedSalesReporting
;
/* Drop table project.FactQueueReporting */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'project.FactQueueReporting') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE project.FactQueueReporting
;


/* Drop table project.DimConsolidatedOrder */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'project.DimConsolidatedOrder') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE project.DimConsolidatedOrder
;
/* Create table project.DimConsolidatedOrder */
CREATE TABLE project.DimConsolidatedOrder (
   [OrderKey]  int IDENTITY  NOT NULL
,  [OrderID]  int   NOT NULL
,  [OrderDateKey]  int   NOT NULL
,  [ShippedDateKey]  int   NOT NULL
,  [Source]  int NULL
,  [RowIsCurrent]  bit   NOT NULL
,  [RowStartDate]  datetime
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)
, CONSTRAINT [PK_project.DimConsolidatedOrder] PRIMARY KEY CLUSTERED
( [OrderKey] )
) ON [PRIMARY]
;
SET IDENTITY_INSERT project.DimConsolidatedOrder ON
;
INSERT INTO project.DimConsolidatedOrder (OrderKey, OrderID, OrderDateKey, ShippedDateKey, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, -1, -1, 0, 12/31/1899, 12/31/9999, 'N/A')
;
SET IDENTITY_INSERT project.DimConsolidatedOrder OFF
;
/* Drop table project.DimConsolidatedProduct */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'project.DimConsolidatedProduct') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE project.DimConsolidatedProduct
;
/* Create table project.DimConsolidatedProduct */
CREATE TABLE project.DimConsolidatedProduct (
   [ProductKey]  int IDENTITY  NOT NULL
,  [ProductID]  int   NOT NULL
,  [ProductName]  varchar(50)   NOT NULL
,  [Department]  nvarchar(50)   NOT NULL
,  [RetailPrice]  money   NOT NULL
,  [WholesalePrice]  money   NOT NULL
,  [VendorName]  varchar(50)   NOT NULL
,  [Source]  int  NULL
,  [IsActive]  bit   NOT NULL
,  [RowIsCurrent]  bit   NOT NULL
,  [RowStartDate]  datetime
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)
, CONSTRAINT [PK_project.DimConsolidatedProduct] PRIMARY KEY CLUSTERED
( [ProductKey] )
) ON [PRIMARY]
;
SET IDENTITY_INSERT project.DimConsolidatedProduct ON
;
INSERT INTO project.DimConsolidatedProduct (ProductKey, ProductID, ProductName, Department, RetailPrice, WholesalePrice, VendorName, Source, IsActive, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'No Product Name', 'No Department', 0, 0, 'No Product Vendor Name', NULL, 0, 0, 12/31/1899, 12/31/9999, 'N/A')
;
SET IDENTITY_INSERT project.DimConsolidatedProduct OFF
;
/* Drop table project.DimMovieTitles */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'project.DimMovieTitles') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE project.DimMovieTitles
;
/* Create table project.DimMovieTitles */
CREATE TABLE project.DimMovieTitles (
   [TitleKey]  int IDENTITY  NOT NULL
,  [TitleID]  varchar(5)   NOT NULL
,  [TitleName]  varchar(200)   NOT NULL
,  [TitleType]  varchar(200)   NOT NULL
,  [TitleSynopsis]  varchar(3000)   NOT NULL
,  [TitleAvgRating]  decimal   NOT NULL
,  [TitleReleaseYear]  int   NOT NULL
,  [TitleRuntime]  int   NOT NULL
,  [TitleRating]  varchar(20)   NOT NULL
,  [TitleBluRayAvailable]  bit   NOT NULL
,  [TitleDvdAvailable]  bit   NOT NULL
,  [TitleInstantAvailable]  bit   NOT NULL
,  [RowIsCurrent]  bit   NOT NULL
,  [RowStartDate]  datetime
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)
, CONSTRAINT [PK_project.DimMovieTitles] PRIMARY KEY CLUSTERED
( [TitleKey] )
) ON [PRIMARY]
;
SET IDENTITY_INSERT project.DimMovieTitles ON
;
INSERT INTO project.DimMovieTitles (TitleKey, TitleID, TitleName, TitleType, TitleSynopsis, TitleAvgRating, TitleReleaseYear, TitleRuntime, TitleRating, TitleBluRayAvailable, TitleDvdAvailable, TitleInstantAvailable, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'Unknown Title', 'Unkown Type', 'Description Unavailable', 0, 0, 0, 'Unknown Rating', 0, 0, 0, 0, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT project.DimMovieTitles OFF
;
/* Drop table project.DimConsolidatedCustomer */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'project.DimConsolidatedCustomer') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE project.DimConsolidatedCustomer
;
/* Create table project.DimConsolidatedCustomer */
CREATE TABLE project.DimConsolidatedCustomer (
   [CustomerKey]  int IDENTITY  NOT NULL
,  [CustomerID]  int   NOT NULL
,  [Source]  int   NOT NULL
,  [FirstName]  varchar(20)   NOT NULL
,  [LastName]  varchar(20)   NOT NULL
,  [FullName]  nvarchar(40) NOT NULL
,  [Address]  nvarchar(50)   NOT NULL
,  [City]  nvarchar(50)   NULL
,  [ZipCode]  varchar(10)   NOT NULL
,  [EmailAddress]  nvarchar(50)   NULL
,  [PhoneNumber]  varchar(50)   NULL
,  [RowIsCurrent]  bit   NOT NULL
,  [RowStartDate]  datetime
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)
, CONSTRAINT [PK_project.DimConsolidatedCustomer] PRIMARY KEY CLUSTERED
( [CustomerKey] )
) ON [PRIMARY]
;
SET IDENTITY_INSERT project.DimConsolidatedCustomer ON
;
INSERT INTO project.DimConsolidatedCustomer (CustomerKey, CustomerID, Source, FirstName, LastName, FullName, Address, City, ZipCode, EmailAddress, PhoneNumber, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, -1, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 0, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT project.DimConsolidatedCustomer OFF
/* Drop table project.DimConsolidatedCustomer */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'project.DimSource') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE project.DimSource
;
/* Create table project.DimConsolidatedCustomer */
CREATE TABLE project.DimSource (
   [SourceKey]  int IDENTITY  NOT NULL
,  [SourceName]  varchar(20)   NOT NULL
, CONSTRAINT [PK_project.DimSource] PRIMARY KEY CLUSTERED
( [SourceKey] )
) ON [PRIMARY]
;
SET IDENTITY_INSERT project.DimSource ON
;
INSERT INTO project.DimSource (SourceKey, SourceName)
VALUES (-1, 'Unknown'), (1, 'FudgeMart'), (2, 'FudgeFlix')
;
SET IDENTITY_INSERT project.DimSource OFF
;
/* Drop table project.DimDate */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'project.DimDate') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE project.DimDate
;
/* Create table project.DimDate */
CREATE TABLE project.DimDate (
   [DateKey]  int   NOT NULL
,  [Date]  date   NULL
,  [FullDateUSA]  nchar(11)   NOT NULL
,  [DayOfWeek]  tinyint   NOT NULL
,  [DayName]  nchar(10)   NOT NULL
,  [DayOfMonth]  tinyint   NOT NULL
,  [DayOfYear]  smallint   NOT NULL
,  [WeekOfYear]  tinyint   NOT NULL
,  [MonthName]  nchar(10)   NOT NULL
,  [MonthOfYear]  tinyint   NOT NULL
,  [Quarter]  tinyint   NOT NULL
,  [QuarterName]  nchar(10)   NOT NULL
,  [Year]  smallint   NOT NULL
,  [IsWeekday]  bit  DEFAULT 0 NOT NULL
, CONSTRAINT [PK_project.DimDate] PRIMARY KEY CLUSTERED
( [DateKey] )
) ON [PRIMARY]
;
INSERT INTO project.DimDate (DateKey, Date, FullDateUSA, DayOfWeek, DayName, DayOfMonth, DayOfYear, WeekOfYear, MonthName, MonthOfYear, Quarter, QuarterName, Year, IsWeekday)
VALUES (-1, '', 'Unk date', 0, 'Unk date', 0, 0, 0, 'Unk month', 0, 0, 'Unk qtr', 0, 0)
;
/* Create table project.FactQueueReporting */
CREATE TABLE project.FactQueueReporting (
   [AccountKey]  int NOT NULL
,  [TitleKey]  int NOT NULL
,  [QueuedDateKey]  int NOT NULL
,  [TimeToShip] int
, CONSTRAINT [PK_project.FactQueueReporting] PRIMARY KEY NONCLUSTERED
( [AccountKey], [TitleKey], [QueuedDateKey] )
) ON [PRIMARY]
;
/* Create table project.FactConsolidatedSalesReporting */
CREATE TABLE project.FactConsolidatedSalesReporting (
   [OrderKey]  int   NOT NULL
,  [CustomerKey]  int   NOT NULL
,  [ProductKey]  int   NOT NULL
,  [InsertAuditKey]  int NOT NULL
,  [UpdateAuditKey]  int NOT NULL
,  [RetailPrice]  money   NOT NULL
,  [Quantity]  int   NOT NULL
,  [OrderTotal]  money   NOT NULL
, CONSTRAINT [PK_project.FactConsolidatedSalesReporting] PRIMARY KEY NONCLUSTERED
( [OrderKey], [ProductKey] )
) ON [PRIMARY]
;
ALTER TABLE project.FactQueueReporting ADD CONSTRAINT
   FK_project_FactQueueReporting_AccountKey FOREIGN KEY
   (
   AccountKey
   ) REFERENCES project.DimConsolidatedCustomer
   ( CustomerKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE project.FactQueueReporting ADD CONSTRAINT
   FK_project_FactQueueReporting_TitleKey FOREIGN KEY
   (
   TitleKey
   ) REFERENCES project.DimMovieTitles
   ( TitleKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE project.FactQueueReporting ADD CONSTRAINT
   FK_project_FactQueueReporting_QueuedDateKey FOREIGN KEY
   (
   QueuedDateKey
   ) REFERENCES project.DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE project.FactConsolidatedSalesReporting ADD CONSTRAINT
   FK_project_FactConsolidatedSalesReporting_OrderKey FOREIGN KEY
   (
   OrderKey
   ) REFERENCES project.DimConsolidatedOrder
   ( OrderKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE project.FactConsolidatedSalesReporting ADD CONSTRAINT
   FK_project_FactConsolidatedSalesReporting_CustomerKey FOREIGN KEY
   (
   CustomerKey
   ) REFERENCES project.DimConsolidatedCustomer
   ( CustomerKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE project.FactConsolidatedSalesReporting ADD CONSTRAINT
   FK_project_FactConsolidatedSalesReporting_ProductKey FOREIGN KEY
   (
   ProductKey
   ) REFERENCES project.DimConsolidatedProduct
   ( ProductKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE project.DimConsolidatedOrder ADD CONSTRAINT
   FK_project_DimConsolidatedOrder_OrderDateKey FOREIGN KEY
   (
   OrderDateKey
   ) REFERENCES project.DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE project.DimConsolidatedOrder ADD CONSTRAINT
   FK_project_DimConsolidatedOrder_ShippedDateKey FOREIGN KEY
   (
   ShippedDateKey
   ) REFERENCES project.DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
/* RK 5/27 */
ALTER TABLE [project].[DimDate] alter column  [Year] [int] NULL
GO
ALTER TABLE [project].[DimDate] alter column  [Date] [datetime] NOT NULL
GO
ALTER TABLE [project].[DimDate] alter column  [DayOfYear] [int] NOT NULL
GO
Alter table [project].[FactConsolidatedSalesReporting]
  add [OrderDateKey] [int] NOT NULL
Alter table [project].[FactConsolidatedSalesReporting]
  alter column [OrderTotal][decimal](25, 4) NOT NULL

ALTER TABLE [project].[FactConsolidatedSalesReporting]
drop column [InsertAuditKey]
ALTER TABLE [project].[FactConsolidatedSalesReporting]
drop column [UpdateAuditKey]

ALTER TABLE [project].[DimConsolidatedCustomer] ADD [State] [nvarchar](2) NULL