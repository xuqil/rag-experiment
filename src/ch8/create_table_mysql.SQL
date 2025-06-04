-- 创建 customers 表

CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20)
);

-- 创建 orders 表

CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2) NOT NULL,
    customer_id INT,
    -- 创建外键约束，关联到 customers 表的 customer_id
    CONSTRAINT fk_orders_customers
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
        ON DELETE CASCADE
);

-- 向 customers 表插入测试数据

INSERT INTO customers (first_name, last_name, email, phone)
VALUES
    ('John', 'Doe', 'john.doe@example.com', '123 - 456 - 7890'),
    ('Jane', 'Smith', 'jane.smith@example.com', '234 - 567 - 8901'),
    ('Bob', 'Johnson', 'bob.johnson@example.com', '345 - 678 - 9012');

-- 向 orders 表插入测试数据

INSERT
	INTO
	orders (total_amount, customer_id)
VALUES
    (150.00,1),
    (200.50,2),
    (75.25,1),
    (300.75,3);
    