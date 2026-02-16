# AUV Interfaces (auv_interfaces)

The `auv_interfaces` package defines the custom ROS 2 messages (`.msg`) and services (`.srv`) used throughout the Nautronics AUV project.

## Overview

This package ensures consistent communication types between different nodes, such as the control system, vision stack, and hardware drivers.

## Usage

To use these interfaces in your Python node:

```python
from auv_interfaces.msg import CustomMessage
from auv_interfaces.srv import CustomService
```

Ensure that your `package.xml` depends on `auv_interfaces`:

```xml
<depend>auv_interfaces</depend>
```
